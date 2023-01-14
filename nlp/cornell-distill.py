# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

# from __future__ import absolute_import, division, print_function
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch_xla
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch.nn.functional as F

from dataclasses import dataclass
from seqeval.metrics import precision_score, recall_score, f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file
from fileutil import Platform
import perm_utils

from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer

class MyDataset(Dataset):
    def __init__(self,all_input_ids, all_input_mask, all_segment_ids, all_labels):
        super(MyDataset, self).__init__()
        self.all_input_ids = all_input_ids
        self.all_input_mask = all_input_mask
        self.all_segment_ids =all_segment_ids
        self.all_labels = all_labels
    def __getitem__(self, index):
        input_ids = self.all_input_ids[index]
        input_mask = self.all_input_mask[index]
        segment_ids = self.all_segment_ids[index]
        labels = self.all_labels[index]
        return {'input_ids':input_ids,
                'attention_mask':input_mask,
                'token_type_ids':segment_ids,
                'labels':labels}
    def __len__(self):
        return len(self.all_labels)

logger = logging.getLogger(__name__)

from transformers import BERT_PRETRAINED_CONFIG_ARCHIVE_MAP
ALL_MODELS = tuple(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())


MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
}

class TPUGeneralDistiller(GeneralDistiller):
    def __init__(self, train_config,
             distill_config,
             model_T,
             model_S,
             adaptor_T,
             adaptor_S,
             permute_logits,
             sampler,
             custom_matches = None):
        # custom_matches=[{'module_T': module_T, 'module_S':module_S,
        #                 'loss': loss, 'weight': weight},...]
        super(TPUGeneralDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)
        self.permute_logits = permute_logits

    def train_with_num_epochs(self, optimizer, scheduler, tqdm_disable, dataloader, max_grad_norm, num_epochs, callback, batch_postprocessor, **args):
        device = xm.xla_device()
        train_steps_per_epoch = len(dataloader)//self.t_config.gradient_accumulation_steps
        total_global_steps = train_steps_per_epoch * num_epochs
        print_every = train_steps_per_epoch // self.print_freq
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [int(train_steps_per_epoch*ci/self.t_config.ckpt_frequency) for ci in range(self.t_config.ckpt_frequency)]

        global_step = 0
        writer_step = 0

        if self.d_config.is_caching_logits is True:
            for step, batch in tqdm(enumerate(dataloader),disable=True):
                self.cache_logits(batch, args, batch_postprocessor)

        for current_epoch in range(int(num_epochs)):
            self.sampler.set_epoch(current_epoch)
            optimizer.zero_grad()
            if self.d_config.is_caching_logits:
                random.shuffle(self.logits_cache)
                dataloader = self.logits_cache

            parallel_loader = pl.ParallelLoader(
                dataloader, [device]).per_device_loader(device)

            for step, batch in enumerate(parallel_loader):
                if self.d_config.is_caching_logits is False and batch_postprocessor is not None:
                        batch = batch_postprocessor(batch)

                total_loss, losses_dict = self.train_on_batch(batch,args)
                # print(losses_dict)
                # self.write_loss(total_loss, writer_step, losses_dict)
                writer_step += 1
                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                if xm.is_master_ordinal() and step % 10 == 0:
                    print("epoch:{} step:{}/{} loss:{}".format(current_epoch, step,
                        len(parallel_loader), total_loss.item()))

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    xm.mark_step()

                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = \
                            self.d_config.kd_loss_weight_scheduler(global_step/total_global_steps)
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = \
                            self.d_config.hard_label_weight_scheduler(global_step/total_global_steps)


            if (current_epoch + 1) % 10 == 0:
                self.save_and_callback(global_step, step, current_epoch, callback)

    def compute_loss(self,results_S,results_T):

        losses_dict = dict()

        total_loss  = 0
        if 'logits' in results_T and 'logits' in results_S:
            logits_list_T = results_T['logits']  # list of tensor
            logits_list_S = results_S['logits']  # list of tensor
            total_kd_loss = 0

            if self.permute_logits:
                xm.mark_step()
                assert len(logits_list_T) == 1
                teacher_logits = logits_list_T[0]

                batch_size = teacher_logits.size(0)
                seq_len = teacher_logits.size(1)
                num_labels = teacher_logits.size(2)
                teacher_targets = torch.argmax(teacher_logits, dim=-1)
                # teacher_targets = results_T["labels"][0]
                # teacher_targets = torch.maximum(torch.tensor(0), teacher_targets)
                # Appear to make things run faster.
                xm.mark_step()
                device = xm.xla_device()
                perm_mat = perm_utils.batch_permutation_matrix(batch_size * seq_len,
                    num_labels, teacher_targets.view(batch_size * seq_len).clone().cpu()).to(device)

                permuted_teacher_logits = torch.matmul(teacher_logits.view(batch_size * seq_len, 1, num_labels),
                    perm_mat.float().to(device)).view(batch_size, seq_len, num_labels)

                logits_list_T = (permuted_teacher_logits,)

            # if 'logits_mask' in results_S:
            #     masks_list_S = results_S['logits_mask']
            #     logits_list_S = select_logits_with_mask(logits_list_S,masks_list_S)  #(mask_sum, num_of_class)
            # if 'logits_mask' in results_T:
            #     masks_list_T = results_T['logits_mask']
            #     logits_list_T = select_logits_with_mask(logits_list_T,masks_list_T)  #(mask_sum, num_of_class)


            if self.d_config.probability_shift is True:
                labels_list = results_S['labels']
                for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                    print("should not hit")
                    l_T = probability_shift_(l_T, labels)
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    total_kd_loss += self.kd_loss(l_S, l_T, temperature)
            else:
                for l_T,l_S in zip(logits_list_T,logits_list_S):
                    if "logits_mask" in results_S:
                        mask = results_S['logits_mask']
                    else:
                        mask = None
                    if self.d_config.temperature_scheduler is not None:
                        temperature = self.d_config.temperature_scheduler(l_S, l_T, self.d_config.temperature)
                    else:
                        temperature = self.d_config.temperature
                    total_kd_loss += self.masked_kd_loss(l_S, l_T, temperature, mask)
            total_loss += total_kd_loss * self.d_config.kd_loss_weight
            losses_dict['unweighted_kd_loss'] = total_kd_loss

        FEATURES = ['hidden','attention']
        inters_T = {feature: results_T.get(feature,[]) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature,[]) for feature in FEATURES}
        inputs_mask_T = results_T.get('inputs_mask',None)
        inputs_mask_S = results_S.get('inputs_mask',None)
        for ith,inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            match_loss = MATCH_LOSS_MAP[loss_type]

            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                name_S = '-'.join(map(str,layer_S))
                name_T = '-'.join(map(str,layer_T))
                if self.projs[ith]:
                    #inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                name_S = str(layer_S)
                name_T = str(layer_T)
                if self.projs[ith]:
                    #inter_T = self.projs[ith](inter_T)
                    inter_S = self.projs[ith](inter_S)
            intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)
            total_loss += intermediate_loss * match_weight
            losses_dict[f'unweighted_{feature}_{loss_type}_{name_S}_{name_T}'] = intermediate_loss

        if self.has_custom_matches:
            for hook_T, hook_S, match_weight, match_loss, proj_func  in \
                    zip(self.custom_matches_cache['hook_outputs_T'], self.custom_matches_cache['hook_outputs_S'],
                        self.custom_matches_cache['match_weghts'], self.custom_matches_cache['match_losses'],
                        self.custom_matches_cache['match_proj_funcs']):
                if proj_func is not None:
                    hook_S = proj_func(hook_S)
                total_loss += match_weight * match_loss(hook_S,hook_T,inputs_mask_S,inputs_mask_T)
            self.custom_matches_cache['hook_outputs_T'] = []
            self.custom_matches_cache['hook_outputs_S'] = []

        if 'losses' in results_S:
            total_hl_loss = 0
            for loss in results_S['losses']:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict['unweighted_hard_label_loss'] = total_hl_loss
        return total_loss, losses_dict

    def masked_kd_loss(self, logits_S, logits_T, temperature=1, mask=None):
        '''
        Calculate the cross entropy between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        assert len(mask) == 1
        mask = mask[0].unsqueeze(-1)

        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1) * mask).sum(dim=-1)
        loss = loss.sum() / torch.sum(mask)
        return loss

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def train(args, train_dataset,model_T, model, tokenizer, labels, pad_token_label_id,predict_callback):
    """ Train the model """
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=args.seed)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_gpu_train_batch_size,
        sampler=train_sampler,
        num_workers=8,
        drop_last=False)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler_class = get_linear_schedule_with_warmup
    scheduler_args = {'num_warmup_steps':int(args.warmup_steps*t_total), 'num_training_steps':t_total}

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    if args.do_train and args.do_distill:
        distill_config = DistillationConfig(
            temperature = args.temperature,
              # intermediate_matches = [{'layer_T':10, 'layer_S':3, 'feature':'hidden','loss': 'hidden_mse', 'weight' : 1}]
            )
        train_config = TrainingConfig(device=args.device,
            log_dir=None,
            output_dir=args.output_dir)
        def adaptor_T(batch,model_output):
            return {"logits":(model_output[1],),
                    'logits_mask':(batch['attention_mask'],),
                    'labels':(batch['labels'],)}
        def adaptor_S(batch,model_output):
            return {"logits":(model_output[1],),
                    'logits_mask':(batch['attention_mask'],),
                    'labels':(batch['labels'],)}

        distiller=TPUGeneralDistiller(train_config,distill_config,
            model_T,model,adaptor_T,adaptor_S,
            permute_logits=args.permute_logits,
            sampler=train_sampler)
        distiller.train(optimizer,train_dataloader,args.num_train_epochs,
                        scheduler_class=scheduler_class, scheduler_args=scheduler_args,
                        max_grad_norm=1.0, callback=predict_callback)
        return


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)
    device = xm.xla_device()

    eval_sampler = torch.utils.data.distributed.DistributedSampler(
        eval_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=42)

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_train_batch_size,
        sampler=eval_sampler,
        num_workers=8,
        drop_last=False)
    eval_dataloader = pl.ParallelLoader(
        eval_dataloader, [device]).per_device_loader(device)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    # eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.per_gpu_train_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, "cached_{}_{}_{}".format(mode,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(examples, labels, args.max_seq_length, tokenizer,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                # xlnet has a cls token at the end
                                                cls_token=tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                sep_token_extra=bool(args.model_type in ["roberta"]),
                                                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                pad_on_left=bool(args.model_type in ["xlnet"]),
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                                                pad_token_label_id=pad_token_label_id
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    if mode=="train":
        dataset = MyDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids) #distill must input a dict
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def _mp_fn(index, args):
    args.device = xm.xla_device()

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO if xm.is_master_ordinal() else logging.WARN)

    # Set seed
    set_seed(args)

    # Prepare CONLL-2003 task
    labels = get_labels(args.labels)
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if not xm.is_master_ordinal():
        xm.rendezvous("dataset_collection")

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if xm.is_master_ordinal():
        if Platform.exists(args.model_name_or_path):
            teacher_dir = os.path.join(os.getcwd(), os.path.basename(args.model_name_or_path))
            Platform.copytree(args.model_name_or_path, teacher_dir)
            logger.info(f"Models downloaded from GCP Bucket to {teacher_dir}.")
            args.model_name_or_path = teacher_dir
        else:
            print("Cannot find teacher directory.")
            exit(1)

    xm.rendezvous("Teacher model downloading.")
    config = config_class.from_pretrained(teacher_dir,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(teacher_dir,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model_T = model_class.from_pretrained(teacher_dir,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    if args.model_name_or_path_student != None:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path_student,
                                              num_labels=num_labels,
                                              cache_dir=args.cache_dir if args.cache_dir else None)
        config.num_hidden_layers=args.num_hidden_layers
        model = model_class.from_pretrained(args.model_name_or_path_student,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        config = config_class.from_pretrained(args.config_name if args.config_name else teacher_dir,
                                          num_labels=num_labels,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
        config.num_hidden_layers=args.num_hidden_layers
        model = model_class.from_pretrained(teacher_dir,
                                        from_tf=bool(".ckpt" in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)


    if xm.is_master_ordinal():
        xm.rendezvous("dataset_collection")

    model.to(args.device)
    model_T.to(args.device)

    # # Evaluating teacher
    # logger.info("Evaluating teacher...")
    # labels = get_labels(args.labels)
    # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    # pad_token_label_id = CrossEntropyLoss().ignore_index
    # tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    # evaluate(args, model_T, tokenizer, labels, pad_token_label_id, mode="test")

    logger.info("Training/evaluation parameters %s", args)
    def predict_callback(model,step):
        labels = get_labels(args.labels)
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = CrossEntropyLoss().ignore_index
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
        model.train()

    # Training
    if args.do_train :
        train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
        # global_step, tr_loss = \
        train(args, train_dataset,model_T, model, tokenizer, labels, pad_token_label_id,predict_callback)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and xm.is_master_ordinal():
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
        model_to_save.cpu().save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    xm.rendezvous("save_model")

    # Evaluation
    results = {}
    if args.do_eval:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test", prefix=global_step)
            if global_step:
                result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
            results.update(result)
        output_eval_file = os.path.join(args.output_dir, "eval_results.pt")
        torch.save(results, output_eval_file)

    xm.rendezvous("upload_model")
    Platform.copytree(args.output_dir, "gs://tianjin-distgen/tjin/" + args.output_dir)

    xm.rendezvous("finish")
    return results

@dataclass
class Options:
    temperature : int
    data_dir : str
    model_type : str
    model_name_or_path : str
    output_dir : str
    num_hidden_layers : int
    nprocs : int
    permute_logits : int

    model_name_or_path_student : str = None
    labels : str = None
    config_name : str = None
    tokenizer_name : str = None
    cache_dir : str = None
    max_seq_length : int = 128
    do_train : bool = True
    do_distill : bool = True
    do_eval : bool = True
    do_predict : bool = True
    evaluate_during_training : bool = True
    do_lower_case : bool = False
    per_gpu_train_batch_size : int = 8
    per_gpu_eval_batch_size : int = 8
    gradient_accumulation_steps : int = 1
    learning_rate : float = 5e-5
    weight_decay : float = 0.0
    adam_epsilon : float = 1e-8
    max_grad_norm : float = 1.0
    num_train_epochs : float = 4.0
    max_steps : int = -1
    warmup_steps : int = 0
    logging_steps : int = 50
    save_steps : int = 50
    eval_all_checkpoints : bool = True
    no_cuda : bool = False
    overwrite_output_dir : bool = True
    overwrite_cache : bool = True
    seed : int = 42
    fp16 : bool = False
    fp16_opt_level : str = "O1"
    local_rank : int = - 1

if __name__ == "__main__":
    from simple_parsing import ArgumentParser
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options

    if Platform.exists(f"gs://tianjin-distgen/tjin/{args.output_dir}/eval_results.pt"):
        print("Already done, quitting.")
        exit(0)

    xmp.spawn(_mp_fn, nprocs=args.nprocs, args=[args])

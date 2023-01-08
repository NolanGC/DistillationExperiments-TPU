import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch

from fileutil import Platform
from torch.utils.data import DataLoader, RandomSampler
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset,load_metric

class TPUGeneralDistiller(GeneralDistiller):
    def __init__(self, train_config,
             distill_config,
             model_T,
             model_S,
             adaptor_T,
             adaptor_S,
             custom_matches = None):
        # custom_matches=[{'module_T': module_T, 'module_S':module_S,
        #                 'loss': loss, 'weight': weight},...]
        super(TPUGeneralDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)

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
            if self.local_rank != -1 and hasattr(dataloader,'sampler'):
                print("set epoch")
                dataloader.sampler.set_epoch(current_epoch)  #In distributed mode, calling the set_epoch method is needed to make shuffling work;
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
                self.write_loss(total_loss, writer_step, losses_dict)
                writer_step += 1
                total_loss /= self.t_config.gradient_accumulation_steps
                if self.t_config.fp16:
                    with amp.scale_loss(total_loss,optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    total_loss.backward()

                if xm.is_master_ordinal() and step % 10 == 0:
                    print("epoch:{} step:{} loss:{}".format(current_epoch, step, total_loss.item()))

                if (step+1)%self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
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


                    if (global_step%train_steps_per_epoch in checkpoints) \
                            and ((current_epoch+1)%self.t_config.ckpt_epoch_frequency==0 or current_epoch+1==num_epochs):
                        self.save_and_callback(global_step, step, current_epoch, callback)

def _mp_fn(index):
    device = xm.xla_device()

    train_dataset = load_dataset('glue', 'sst2', split='train')
    val_dataset = load_dataset('glue', 'sst2', split='validation')
    test_dataset = load_dataset('glue', 'sst2', split='test')

    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)

    val_dataset = val_dataset.remove_columns(['label'])
    test_dataset = test_dataset.remove_columns(['label'])
    train_dataset = train_dataset.remove_columns(['label'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LENGTH = 128
    train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        seed=42)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=4,
        drop_last=False)

    bert_config_T3 = BertConfig.from_json_file('./configs/bert_config_L3.json')
    bert_config_T3.output_hidden_states = True

    student_model = BertForSequenceClassification(bert_config_T3) #, num_labels = 2
    student_model.to(device=device)

    bert_config = BertConfig.from_json_file('./configs/bert_config.json')
    bert_config.output_hidden_states = True
    teacher_model = BertForSequenceClassification(bert_config) #, num_labels = 2
    teacher_model.load_state_dict(Platform.load_model('gs://tianjin-distgen/sst2_teacher_model.pt', map_location=torch.device('cpu')))
    teacher_model.to(device=device)

    num_epochs = 20
    num_training_steps = len(train_loader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

    distill_config = DistillationConfig(
        is_caching_logits=False,
        intermediate_matches=[
         {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
         {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])
    train_config = TrainingConfig(device=device)

    distiller = TPUGeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


    with distiller:
        distiller.train(optimizer, train_loader, num_epochs,
            scheduler_class=scheduler_class,
            scheduler_args=scheduler_args,
            callback=None)

    xm.rendezvous("training_end")

if __name__ == '__main__':
    xmp.spawn(_mp_fn, nprocs=8, args=())
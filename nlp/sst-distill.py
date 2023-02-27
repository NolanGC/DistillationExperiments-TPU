import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch
import os

from fileutil import Platform
from torch.utils.data import DataLoader, RandomSampler
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from distiller import TPUGeneralDistiller
from dataclasses import dataclass

from simple_parsing import ArgumentParser

def save(model, tokenizer, args):
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.cpu().save_pretrained(args.experiment_name)
    tokenizer.save_pretrained(args.experiment_name)
    

def _mp_fn(index, args):
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
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        drop_last=False)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=False,
        seed=42)
    test_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=256,
        sampler=test_sampler,
        num_workers=4,
        drop_last=False)

    bert_config_T3 = BertConfig.from_json_file('./configs/bert_config_L3.json')
    student_model = BertForSequenceClassification(bert_config_T3) #, num_labels = 2
    student_model.to(device=device)

    bert_config = BertConfig.from_json_file('./configs/bert_config.json')
    bert_config.output_hidden_states = True
    teacher_model = BertForSequenceClassification(bert_config) #, num_labels = 2
    teacher_model.load_state_dict(Platform.load_model('gs://tianjin-distgen/sst2_teacher_model.pt', map_location=torch.device('cpu')))
    teacher_model.to(device=device)

    num_epochs = 1
    num_training_steps = len(train_loader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=(0.1*num_training_steps), 
                                                num_training_steps=num_training_steps)

    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

    distill_config = DistillationConfig(
        is_caching_logits=False,
        temperature = args.temperature)
    train_config = TrainingConfig(device=device)

    distiller = TPUGeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model,
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor,
        sampler=train_sampler)

    distiller.train(optimizer=optimizer, scheduler=scheduler, dataloader=train_loader, num_epochs=5)
    
    xm.rendezvous("upload_model")
    Platform.copytree(args.experiment_name, 
                      os.path.join("gs://tianjin-distgen/tjin/nlp", args.experiment_name))

    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)
    student_model.eval()

    print("eval")
    device = xm.xla_device()
    parallel_loader = pl.ParallelLoader(
        train_loader, [device]).per_device_loader(device)
    for batch in parallel_loader:
        batch = {k: v for k, v in batch.items()}
        with torch.no_grad():
            outputs = student_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        correct += torch.sum(torch.eq(predictions, batch["labels"]))
        total += torch.tensor(predictions.size(0))
        xm.mark_step()


    print(correct, total)
    print(correct/total.cpu().item())

    xm.rendezvous("training_end")

@dataclass
class Options:
    experiment_name : str 
    temperature : int = 4

    # data_dir : str
    # model_type : str
    # model_name_or_paths : str
    # output_dir : str
    # num_hidden_layers : int
    # nprocs : int
    # permute_logits : int

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(_mp_fn, nprocs=1, args=(args,))
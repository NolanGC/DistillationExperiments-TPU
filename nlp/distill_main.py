import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch
import os
import wandb

from fileutil import Platform
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup
from distiller import TPUGeneralDistiller
from dataclasses import dataclass
from dataset import get_dataloader, DatasetKind, PartitionKind, DataOption
from pydantic.dataclasses import dataclass
from simple_parsing import ArgumentParser
from misc import silence, flatten

CWD = os.path.dirname(os.path.abspath(__file__))

def save(model, tokenizer, args):
    model_to_save = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_save.cpu().save_pretrained(args.experiment_name)
    tokenizer.save_pretrained(args.experiment_name)

def _mp_fn(index, args):
    device = xm.xla_device()
    
    train_loader = get_dataloader(DatasetKind.SST2, PartitionKind.TRAIN, args.data, silence=True)
    test_loader = get_dataloader(DatasetKind.SST2, PartitionKind.TEST, args.data, silence=True)

    with silence(enable=True):
        config = BertConfig.from_pretrained('bert-base-uncased', num_labels=2, cache_dir=None)
        config.num_hidden_layers = 3
        config.num_labels = 2
        student_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', from_tf=False, config=config, cache_dir=None)
        student_model.to(device=device)

        bert_config = BertConfig.from_json_file(os.path.join(CWD, 'configs/bert_config.json'))
        bert_config.num_labels = 2
        teacher_model = BertForSequenceClassification(bert_config)
        teacher_model.load_state_dict(Platform.load_model('gs://tianjin-distgen/sst2_teacher_model.pt', map_location=torch.device('cpu')))
        teacher_model.to(device=device)
    
    if xm.is_master_ordinal():
        wandb.init(
            project="sst2-distillation",
            name=args.experiment_name,
            config=flatten(vars(args)))
    
    num_training_steps = len(train_loader) * args.train.epochs
    optimizer = AdamW(student_model.parameters(), lr=args.train.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, 
                                                num_warmup_steps=(0.1*num_training_steps), 
                                                num_training_steps=num_training_steps)

    distiller = TPUGeneralDistiller(
        model_T=teacher_model, model_S=student_model,
        sampler=train_loader.sampler, train_args=args.train,
        train_loader=train_loader, eval_loader=test_loader)

    distiller.train(optimizer=optimizer, scheduler=scheduler, 
                    num_epochs=args.train.epochs)
    
    xm.rendezvous("upload_model")
    Platform.copytree(args.experiment_name, 
                      os.path.join("gs://tianjin-distgen/tjin/nlp", args.experiment_name))

    correct = torch.tensor(0.0).to(device)
    total = torch.tensor(0.0).to(device)
    student_model.eval()

    print("eval")
    device = xm.xla_device()
    parallel_loader = pl.ParallelLoader(
        test_loader, [device]).per_device_loader(device)
    for batch in parallel_loader:
        with torch.no_grad():
            outputs = student_model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        correct += torch.sum(torch.eq(predictions, batch["labels"]))
        total += torch.tensor(predictions.size(0))
        xm.mark_step()

    correct, total = xm.all_reduce(xm.REDUCE_SUM, [correct, total])
    xm.mark_step()
    xm.master_print(correct, total, correct/total.cpu().item())

    xm.rendezvous("training_end")

@dataclass
class TrainOption:
    eval_freq : int = 5
    lr : float = 1e-4
    epochs : int = 30
    temperature : int = 8
    one_hot : bool = False
    el2n_threshold : float = None

    # By default, difficult (high-EL2N) examples use teacher outputs during training, easy ones use onehot labels.
    # By setting this flag to true the reverse is true -- difficult examples uses onehot labels.
    el2n_invert_filter : bool = False

@dataclass
class Options:
    experiment_name : str 
    train : TrainOption = TrainOption()
    data : DataOption = DataOption()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_arguments(Options, dest="options")
    args = parser.parse_args().options
    xmp.spawn(_mp_fn, nprocs=8, args=(args,))
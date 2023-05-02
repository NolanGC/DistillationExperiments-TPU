import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from fileutil import Platform
import argparse

import torch
device='cuda' if torch.cuda.is_available() else 'cpu'

import os
import torch
from transformers import BertForSequenceClassification, BertTokenizer,BertConfig
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
}

def _mp_fn(index, args):
    train_dataset = load_dataset('glue', 'sst2', split='train')
    val_dataset = load_dataset('glue', 'sst2', split='validation')
    test_dataset = load_dataset('glue', 'sst2', split='test')
    train_dataset = train_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset = val_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    test_dataset = test_dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    val_dataset = val_dataset.remove_columns(['label'])
    test_dataset = test_dataset.remove_columns(['label'])
    train_dataset = train_dataset.remove_columns(['label'])

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
    config = config_class.from_pretrained('bert-base-uncased', num_labels=2, cache_dir=None)
    config.num_hidden_layers = 3
    model = model_class.from_pretrained('bert-base-uncased', from_tf=False, config=config, cache_dir=None)
    # model.load_state_dict(Platform.load_model('gs://tianjin-distgen/sst2_teacher_model.pt', map_location=torch.device('cpu')))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LENGTH = 128
    train_dataset = train_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    val_dataset = val_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    #start training
    training_args = TrainingArguments(
        output_dir=f'./results-{args.id}',          #output directory
        learning_rate=5e-5,
        num_train_epochs=1,
        per_device_train_batch_size=32,                #batch size per device during training
        per_device_eval_batch_size=100,                #batch size for evaluation
        logging_dir=None,
        logging_steps=1000,
        do_train=True,
        do_eval=True,
        no_cuda=False,
        load_best_model_at_end=True,
        evaluation_strategy="epoch",
        seed=args.id,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    xm.rendezvous("training_start")

    train_out = trainer.train()
    eval_output = trainer.evaluation_loop(trainer.get_eval_dataloader(train_dataset), "Evaluation")
    preds, labels = eval_output.predictions, eval_output.label_ids

    # if xm.is_master_ordinal():
    #     Platform.save_model({
    #         "pred": preds,
    #         "labels": labels,
    #     }, f"gs://tianjin-distgen/sst2/el2n-raw-{args.id}.pt")
    # xm.rendezvous("training_end")

    # if xm.is_master_ordinal():
    #     Platform.save_model(model.cpu().state_dict(), 
    #                         f'gs://tianjin-distgen/sst2/el2n-student-1ep-id{args.id}.pt')
    # xm.rendezvous("exit_together")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int)
    args = parser.parse_args()
    xmp.spawn(_mp_fn, nprocs=1, args=(args,))
    

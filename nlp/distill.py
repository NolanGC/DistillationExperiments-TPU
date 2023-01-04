import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch

from fileutil import Platform

from torch.utils.data import DataLoader, RandomSampler
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from transformers import BertForSequenceClassification, BertConfig, AdamW,BertTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset,load_metric

def _mp_fn(index):
    device = xm.xla_device()

    train_dataset = load_dataset('glue', 'sst2', split='train')
    val_dataset = load_dataset('glue', 'sst2', split='validation')
    test_dataset = load_dataset('glue', 'sst2', split='test')

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=32) #prepare dataloader

    bert_config_T3 = BertConfig.from_json_file('./TextBrewer/examples/student_config/bert_base_cased_config/bert_config_L3.json')
    bert_config_T3.output_hidden_states = True

    student_model = BertForSequenceClassification(bert_config_T3) #, num_labels = 2
    student_model.to(device=device)
 

    bert_config = BertConfig.from_json_file('./TextBrewer/examples/student_config/bert_base_cased_config/bert_config.json')
    bert_config.output_hidden_states = True
    teacher_model = BertForSequenceClassification(bert_config) #, num_labels = 2
    teacher_model.load_state_dict(Platform.load_model('gs://tianjin-distgen/sst2_teacher_model.pt', map_location=torch.device('cpu')))
    teacher_model.to(device=device)
    
    num_epochs = 20
    num_training_steps = len(train_dataloader) * num_epochs
    # Optimizer and learning rate scheduler
    optimizer = AdamW(student_model.parameters(), lr=1e-4)

    scheduler_class = get_linear_schedule_with_warmup
    # arguments dict except 'optimizer'
    scheduler_args = {'num_warmup_steps':int(0.1*num_training_steps), 'num_training_steps':num_training_steps}


    def simple_adaptor(batch, model_outputs):
        return {'logits': model_outputs.logits, 'hidden': model_outputs.hidden_states}

    distill_config = DistillationConfig(
        intermediate_matches=[    
         {'layer_T':0, 'layer_S':0, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1},
         {'layer_T':8, 'layer_S':2, 'feature':'hidden', 'loss': 'hidden_mse','weight' : 1}])
    train_config = TrainingConfig(device=device)

    distiller = GeneralDistiller(
        train_config=train_config, distill_config=distill_config,
        model_T=teacher_model, model_S=student_model, 
        adaptor_T=simple_adaptor, adaptor_S=simple_adaptor)


    with distiller:
        distiller.train(optimizer, train_dataloader, num_epochs, 
            scheduler_class=scheduler_class, 
            scheduler_args = scheduler_args, callback=None)

    # xm.rendezvous("training_start")

    # train_out = trainer.train()
    # # print(trainer.evaluate())
    # if xm.is_master_ordinal():
    #     Platform.save_model(model.cpu().state_dict(), 'gs://tianjin-distgen/sst2_teacher_model.pt')


    xm.rendezvous("training_end")

if __name__ == '__main__':
    xmp.spawn(_mp_fn, nprocs=1, args=())
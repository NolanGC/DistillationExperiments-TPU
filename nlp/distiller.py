import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch
import torch.nn.functional as F
from textbrewer import DistillationConfig,TrainingConfig,GeneralDistiller
from misc import masked_kd_loss

def post_adaptor(dict_object):
    if 'logits' in dict_object:
        logits = dict_object['logits']
        if not isinstance(logits,(list,tuple)):
            dict_object['logits'] = [ logits ]
    return dict_object

class TPUGeneralDistiller(GeneralDistiller):
    def __init__(self, train_config,
             distill_config,
             model_T,
             model_S,
             adaptor_T,
             adaptor_S,
             sampler,
             permute_logits=False):

        super(TPUGeneralDistiller, self).__init__(train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S)
        self.permute_logits = permute_logits
        self.sampler = sampler
    
    def train(self, optimizer, scheduler, dataloader, num_epochs=None, max_grad_norm=-1.0, **args):
        device = xm.xla_device()
        global_step = 0
        for current_epoch in range(int(num_epochs)):
            self.sampler.set_epoch(current_epoch)
            optimizer.zero_grad()

            parallel_loader = pl.ParallelLoader(
                dataloader, [device]).per_device_loader(device)

            for step, batch in enumerate(parallel_loader):
                with torch.no_grad():
                    results_T = self.model_T(**batch, **args)
                results_S = self.model_S(**batch, **args)
                teacher_logit = results_T.logits
                student_logit = results_S.logits

                temperature = self.d_config.temperature
                loss = masked_kd_loss(student_logit, teacher_logit, temperature)
                loss = loss * self.d_config.kd_loss_weight
                loss.backward()

                if xm.is_master_ordinal() and step % 10 == 0:
                    print("epoch:{} step:{}/{} loss:{}".format(current_epoch, step,
                        len(parallel_loader), loss.item()))

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_S.parameters(), max_grad_norm)
                
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
import torch
import torch.nn.functional as F
import wandb
from textbrewer import GeneralDistiller
from misc import kd_loss

class TPUGeneralDistiller:
    def __init__(self,
             model_T,
             model_S,
             sampler,
             train_args,
             train_loader,
             eval_loader,
             permute_logits=False):

        self.teacher = model_T
        self.student = model_S
        self.permute_logits = permute_logits
        self.sampler = sampler
        self.train_args = train_args
        self.eval_loader = eval_loader
        self.train_loader = train_loader
    
    def eval(self, global_step):
        device = xm.xla_device()
        correct = torch.tensor(0.0).to(device)
        total = torch.tensor(0.0).to(device)
        self.student.eval()

        device = xm.xla_device()
        parallel_loader = pl.ParallelLoader(
            self.eval_loader, [device]).per_device_loader(device)
        for batch in parallel_loader:
            with torch.no_grad():
                outputs = self.student(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            
            correct += torch.sum(torch.eq(predictions, batch["labels"]))
            total += torch.tensor(predictions.size(0))
            xm.mark_step()

        correct, total = xm.all_reduce(xm.REDUCE_SUM, [correct, total])
        accuracy = correct/total.cpu().item()
        xm.mark_step()

        if xm.is_master_ordinal():
            print(f"[eval] {correct}/{total} {accuracy}")
            wandb.log({'accuracy': accuracy}, step=global_step)

    def train(self, optimizer, scheduler, num_epochs=None, max_grad_norm=-1.0, **args):
        self.teacher.train()
        device = xm.xla_device()
        global_step = 0
        for current_epoch in range(int(num_epochs)):
            self.sampler.set_epoch(current_epoch)
            optimizer.zero_grad()

            parallel_loader = pl.ParallelLoader(
                self.train_loader, [device]).per_device_loader(device)

            for step, batch in enumerate(parallel_loader):
                el2n_scores = batch.pop("el2n")
                with torch.no_grad():
                    results_T = self.teacher(**batch, **args)
                results_S = self.student(**batch, **args)
                teacher_logit = results_T.logits
                student_logit = results_S.logits

                assert self.train_args.one_hot != (self.train_args.el2n_threshold != None), "Cannot specify EL2N threshold while training with one-hot labels."
                labels = batch["labels"]
                if self.train_args.one_hot:
                    teacher_logit = F.one_hot(labels, num_classes=student_logit.shape[-1])
                elif self.train_args.el2n_threshold != None:
                    difficulty = el2n_scores > self.train_args.el2n_threshold
                    onehot_labels = F.one_hot(labels, num_classes=student_logit.shape[-1]).float()
                    teacher_logit = torch.where(torch.unsqueeze(difficulty, dim=-1), teacher_logit, onehot_labels)
                
                
                loss = kd_loss(student_logit, teacher_logit, self.train_args.temperature)
                loss.backward()

                if xm.is_master_ordinal() and step % 10 == 0:
                    print("epoch:{} step:{}/{} loss:{}".format(current_epoch, step,
                        len(parallel_loader), loss.item()))
                    wandb.log({'loss': loss.item(),
                               'lr': optimizer.param_groups[-1]['lr']}, step=global_step)

                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_grad_norm)
                
                xm.optimizer_step(optimizer)
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            
            if (current_epoch + 1) % self.train_args.eval_freq == 0:
                self.eval(global_step - 1)
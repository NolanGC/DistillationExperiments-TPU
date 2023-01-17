import torch
import torch.nn.functional as F
from utils import reduce_ensemble_logits

class ClassifierStudentLoss(object):
    def __init__(self, student_model, base_loss, device=None, alpha=0.9):
        self.student = student_model
        self.device = device
        self.base_loss = base_loss
        self.alpha = alpha

    def __call__(self, inputs, targets, teacher_logits, temp=None):
        real_batch_size = targets.size(0)
        self.student.to(self.device)
        self.student.to(self.device)
        student_logits = self.student(inputs.to(self.device))
        hard_loss = F.cross_entropy(student_logits[:real_batch_size], targets)
        # temp = torch.ones_like(student_logits) if temp is None else temp.unsqueeze(-1)
        temp = torch.ones_like(student_logits) if temp is None else temp
        soft_loss = self.base_loss(teacher_logits, student_logits, temp)
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        return loss, student_logits

class ClassifierTeacherLoss(object):
    def __init__(self, teacher_model, dev):
        self.teacher = teacher_model
        self.device = dev

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs.to(self.device))
        loss = F.cross_entropy(logits.to(self.device), targets.to(self.device))
        return loss, logits

class ClassifierTeacherLossWithTemp(object):
    def __init__(self, teacher_model, dev, temp, num_classes):
        self.teacher = teacher_model
        self.device = dev
        self.temp = temp
        self.num_classes = num_classes

    def __call__(self, inputs, targets):
        logits = self.teacher(inputs.to(self.device))
        teacher_probs = F.one_hot(targets, num_classes=self.num_classes)
        student_logp = F.log_softmax(logits / self.temp, dim=-1)
        loss = -(self.temp ** 2 * teacher_probs * student_logp).sum(-1).mean()
        return loss, logits


class TeacherStudentFwdCrossEntLoss(object):
    #Soft teacher/student cross entropy loss from [Hinton et al (2015)]
     #   (https://arxiv.org/abs/1503.02531)

    def __call__(self, teacher_logits, student_logits, temp):
        if teacher_logits.dim() == 3:
            teacher_logits = reduce_ensemble_logits(teacher_logits)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
        student_logp = F.log_softmax(student_logits / temp, dim=-1)
        loss = -(temp ** 2 * teacher_probs * student_logp).sum(-1).mean()
        return loss

class ClassifierEnsembleLoss(object):
    def __init__(self, ensemble, device):
        self.ensemble = ensemble
        self.device = device

    def __call__(self, inputs, targets):
        logits = self.ensemble(inputs.to(self.device))
        logits = reduce_ensemble_logits(logits)
        return F.nll_loss(logits, targets), logits


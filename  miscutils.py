import torch
import numpy as np
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
import math

def reduce_ensemble_logits(teacher_logits):
    assert teacher_logits.dim() == 3
    teacher_logits = F.log_softmax(teacher_logits, dim=-1)
    n_teachers = len(teacher_logits)
    return torch.logsumexp(teacher_logits, dim=1) - math.log(n_teachers)

def batch_calibration_stats(logits, targets, num_bins):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins).to(logits.device)
    probs, preds = logits.softmax(dim=-1).max(-1)
    bin_correct = torch.zeros(num_bins).float()
    bin_prob = torch.zeros(num_bins).float()
    bin_count = torch.zeros(num_bins).float()
    for idx, conf_level in enumerate(bin_bounds):
        mask = (conf_level - 1 / num_bins < probs) * (probs <= conf_level)
        num_elements = mask.sum().float()
        total_correct = 0. if num_elements < 1 else preds[mask].eq(targets[mask]).sum()
        total_prob = 0. if num_elements < 1 else probs[mask].sum()
        bin_count[idx] = num_elements
        bin_correct[idx] = total_correct
        bin_prob[idx] = total_prob
    return bin_count, bin_correct, bin_prob

def expected_calibration_err(bin_count, bin_correct, bin_prob, num_samples):
    ece = 0
    for count, correct, prob in zip(bin_count, bin_correct, bin_prob):
        if count < 1:
            continue
        ece += count / num_samples * abs(correct / count - prob / count)
    return ece.item()

def ece_bin_metrics(bin_count, bin_correct, bin_prob, num_bins, prefix):
    bin_bounds = torch.linspace(1 / num_bins, 1.0, num_bins)
    assert bin_bounds.size(0) == bin_count.size(0)
    bin_acc = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_correct, bin_count))
    bin_conf = map(lambda x: 0. if x[1] < 1 else (x[0] / x[1]).item(), zip(bin_prob, bin_count))
    metrics = {f"{prefix}_bin_count_{ub:0.2f}": count.item() for ub, count in zip(bin_bounds, bin_count)}
    metrics.update(
        {f"{prefix}_bin_acc_{ub:0.2f}": acc for ub, acc in zip(bin_bounds, bin_acc)}
    )
    metrics.update(
        {f"{prefix}_bin_conf_{ub:0.2f}": conf for ub, conf in zip(bin_bounds, bin_conf)}
    )
    return metrics


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)

def preact_cka(teacher, student, dataloader):
    """
    https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.ipynb
    """
    cka = None
    for inputs, _ in dataloader:
        inputs = try_cuda(inputs)
        with torch.no_grad():
            teacher_preacts = teacher.preacts(inputs)
            student_preacts = student.preacts(inputs)

        assert len(teacher_preacts) == len(student_preacts)
        batch_cka = np.empty((len(teacher_preacts),))
        for idx, (t_preact, s_preact) in enumerate(zip(teacher_preacts, student_preacts)):
            t_preact = t_preact.cpu().numpy()  # [batch_size x resolution x resolution x channel_size]
            s_preact = s_preact.cpu().numpy()  # [batch_size x resolution x resolution x channel_size]
            avg_t_preact = np.mean(t_preact, axis=(1, 2))
            avg_s_preact = np.mean(s_preact, axis=(1, 2))
            batch_cka[idx] = kernel_CKA(avg_t_preact.T, avg_s_preact.T)

        if cka is None:
            cka = batch_cka / len(dataloader)
        else:
            cka += batch_cka / len(dataloader)
    return cka

def save_obj(obj, filename):
    #TODO change to using GCS instead of local file system
    #save_path = filename
    #torch.save(obj, save_path)
    pass
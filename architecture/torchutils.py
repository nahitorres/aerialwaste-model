import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import numpy as np
import ignite.metrics as metrics
#import ignite.contrib.metrics.average_precision as average_precision

def average_precision_compute_fn(y_preds, y_targets):
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        raise RuntimeError("This contrib module requires sklearn to be installed.")
    y_true = y_targets.numpy()
    y_pred = y_preds.numpy()
    a = average_precision_score(y_true, y_pred, average=None)
    res =np.nan_to_num(a)
    return res

class AveragePrecision(metrics.EpochMetric):
    
    def __init__(self, output_transform=lambda x: x):
        super(AveragePrecision, self).__init__(average_precision_compute_fn, output_transform=output_transform)
        self.num_classes = 0

    def reset(self):
        self.num_classes = 0
        super(AveragePrecision, self).reset()
        
    def update(self, output):

        y_pred, y = output
        self._predictions = torch.as_tensor(self._predictions).clone().detach()
        self._targets = torch.as_tensor(self._targets).clone().detach()
        
        self.num_classes = y_pred.size(1)
        y_pred = y_pred.type_as(self._predictions)
        y = y.type_as(self._targets)
        self._predictions = torch.cat([self._predictions, y_pred], dim=0)
        self._targets = torch.cat([self._targets, y], dim=0)

    def compute(self):
        _predictions = self._predictions.reshape(-1, self.num_classes)
        _targets = self._targets.reshape(-1, self.num_classes)
        return self.compute_fn(_predictions, _targets)


class Accuracy(metrics.Accuracy):
    def __init__(self, output_transform=lambda x: x, is_multilabel=True):
        super(Accuracy, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel)
        self._correct = None
        self._examples = None

    def reset(self):
        self._correct = None
        self._examples = None
        super(Accuracy, self).reset()

    def update(self, output):
        
        y_pred, y = output

        num_classes = y_pred.size(1)
        last_dim = y_pred.ndimension()

        y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
        y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
        correct = (y == y_pred.type_as(y))
        correct = correct.sum(dim=0).type(torch.DoubleTensor)

        num_examples = y.shape[0]
        if self._correct is None:
            self._correct = correct
            self._examples = num_examples
        else:
            self._correct += correct
            self._examples += num_examples

    def compute(self):
        return self._correct / self._examples


class Precision(metrics.Precision):
    def __init__(self, output_transform=lambda x: x, average=True, is_multilabel=True):
        super(Precision, self).__init__(output_transform=output_transform, average=average, is_multilabel=is_multilabel)
        self.num_classes = 0

    def update(self, output):
        y_pred, y = output

        num_classes = y_pred.size(1)
        self.num_classes = num_classes
        y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
        y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        y = y.type_as(y_pred)
        correct = y * y_pred
        all_positives = y_pred.sum(dim=1).type(torch.DoubleTensor)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(all_positives)
        else:
            true_positives = correct.sum(dim=1)
        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / all_positives
        true_positives = true_positives.type(torch.DoubleTensor)

        self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
        self._positives = torch.cat([self._positives, all_positives], dim=0)

    def compute(self):
        _true_positives = self._true_positives.reshape(-1, self.num_classes)
        _true_positives = _true_positives.sum(dim=0)
        _positives = self._positives.reshape(-1, self.num_classes)
        _positives = _positives.sum(dim=0)

        return _true_positives / (_positives + self.eps)


class Recall(metrics.Recall):
    def __init__(self, output_transform=lambda x: x, average=True, is_multilabel=True):
        super(Recall, self).__init__(output_transform=output_transform, average=average, is_multilabel=is_multilabel)
        self.num_classes = 0

    def update(self, output):
        y_pred, y = output

        num_classes = y_pred.size(1)
        self.num_classes = num_classes
        y_pred = torch.transpose(y_pred, 1, 0).reshape(num_classes, -1)
        y = torch.transpose(y, 1, 0).reshape(num_classes, -1)

        y = y.type_as(y_pred)
        correct = y * y_pred
        actual_positives = y.sum(dim=1).type(torch.DoubleTensor)  # Convert from int cuda/cpu to double cpu

        if correct.sum() == 0:
            true_positives = torch.zeros_like(actual_positives)
        else:
            true_positives = correct.sum(dim=1)

        # Convert from int cuda/cpu to double cpu
        # We need double precision for the division true_positives / actual_positives
        true_positives = true_positives.type(torch.DoubleTensor)

        self._true_positives = torch.cat([self._true_positives, true_positives], dim=0)
        self._positives = torch.cat([self._positives, actual_positives], dim=0)

    def compute(self):
        _true_positives = self._true_positives.reshape(-1, self.num_classes)
        _true_positives = _true_positives.sum(dim=0)

        _positives = self._positives.reshape(-1, self.num_classes)
        _positives = _positives.sum(dim=0)

        return _true_positives / (_positives + self.eps)


class FocalLoss(nn.Module):
    # object detection
    def __init__(self, gamma=2, balance_param=0.25, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.balance_param = balance_param
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        # compute the negative likelyhood
        input = torch.sigmoid(input)
        logpt = - F.binary_cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(logpt)

        # compute the loss
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        # focal_loss = self.balance_param * focal_loss

        return focal_loss.mean()


def focal_loss(input, target, gamma=2.0, weight=None, reduction='mean'):
    return FocalLoss(gamma, weight=weight, reduction=reduction).forward(input, target)


class PolyAdaDeltaptimizer(torch.optim.Adadelta):

    def __init__(self, params, lr, max_step):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step

    def step(self, closure=None):
        super().step(closure)

        self.global_step += 1


class PolyAdamOptimizer(torch.optim.Adam):

    def __init__(self, params, lr, max_step):
        super().__init__(params, lr)

        self.global_step = 0
        self.max_step = max_step

    def step(self, closure=None):
        super().step(closure)

        self.global_step += 1


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step)) / 2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out

"""Noam Scheduler."""
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
class NoamLR(lr_scheduler._LRScheduler):
    r"""Noam Learning rate schedule.
    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.
              ^
             / \
            /   `
           /     `
          /         `
         /               `
        /                       `
       /                                   `
      /                                                    `
     /                                                                              `
    /                                                                                                                  `
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    Notes
    -----
    If step <= warmup_steps,
        scale = step / warmup_steps
    If step > warmup_steps,
        scale = (warmup_steps ^ 0.5) / (step ^ 0.5)
    """
    def __init__(self, optimizer, model_size, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer)

    def scale(self, step):
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def get_lr(self):
        scale = self.scale(max(1,self._step_count))
        return [base_lr * scale for base_lr in self.base_lrs]
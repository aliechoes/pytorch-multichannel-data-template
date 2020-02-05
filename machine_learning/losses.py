import torch.nn as nn
import torchvision
import torch.nn.functional as F
  
from torch.autograd import Function


def dice_coeff(inputs, target):
    """Dice coeff for batches"""
    if inputs.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(inputs, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, inputs, target):
        self.save_for_backward(inputs, target)
        eps = 0.0001
        self.inter = torch.dot(inputs.view(-1), target.view(-1))
        self.union = torch.sum(inputs) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        inputs, target = self.saved_variables
        grad_inputs = grad_target = None

        if self.needs_inputs_grad[0]:
            grad_inputs = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_inputs_grad[1]:
            grad_target = None

        return grad_inputs, grad_target


def get_loss(loss_function):
    if loss_function=="mse":
        criterion = nn.MSELoss()
        return criterion
    if loss_function== "nll_loss": 
        criterion = nn.NLLLoss()
        return criterion
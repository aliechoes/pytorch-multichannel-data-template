
# Losses

For adding new losses, you can simply add your model in the file [__init__.py](__init__.py).  It should be part of the function `get_loss`. In addition, it should be added to the `config.json` to be considered.

## list of available losses

- [mse](https://pytorch.org/docs/stable/nn.html#mseloss) 
- [nll_loss](https://pytorch.org/docs/stable/nn.html#nllloss)
- [FocalLoss2d](https://arxiv.org/abs/1708.02002)


# Optimizers

For adding new optimizers, you can simply add your model in the file [__init__.py](__init__.py).  It should be part of the function `get_optimizer`. In addition, it should be added to the `config.json` to be considered.

### list of available optimizers

- [adam](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html)
- [sgd](https://pytorch.org/docs/stable/_modules/torch/optim/sgd.html)
- [rmsprop](https://pytorch.org/docs/stable/_modules/torch/optim/rmsprop.html)


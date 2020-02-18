# Machine Learning	

Here you can find the machine learning part of the pipeline. It inlcudes the deap learning `models`, `losses`, `optimization` algorithms as well as the `metrics` which are going to be tracked.

## Models

At the moment, there is a simple deep learning model which can be simply used for test. It is called `ShallowNet`. Apart from that, `AlexNet` is implemented. The difference between a normal architecture with the implemented one is that the implemented ones can detect how many channels are available and initialized the model using that information. While this brings the limitation in case of pretrained models. It means that for models which have more or less than three channels, we cannot use pretrained models. However, it can benefit from the architecture.

In addition, every model includes a part called `embedding_generator` which can be used to extract features right away from the model with no need to tweak anything.

For adding new models, you can simply add your model in the file [models.py](models.py). Please consider that it has to have the method `embedding_generator`. In addition, it should be added to the `config.json` to be considered.

## Metrics

For adding new metrics, you can simply add your model in the file [metrics.py](metrics.py).  It should be part of the function `metric_history`. In addition, it should be added to the `config.json` to be considered. 

### list of available metrics

- [accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)
- [cross_entropy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)
- [dice](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)

## Losses

For adding new losses, you can simply add your model in the file [losses.py](losses.py).  It should be part of the function `get_loss`. In addition, it should be added to the `config.json` to be considered.

### list of available losses

- [mse](https://pytorch.org/docs/stable/nn.html#mseloss) 
- [nll_loss](https://pytorch.org/docs/stable/nn.html#nllloss)

## Optimizers

For adding new optimizers, you can simply add your model in the file [optimizers.py](optimizers.py).  It should be part of the function `get_optimizer`. In addition, it should be added to the `config.json` to be considered.

### list of available optimizers

- [adam](https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam)


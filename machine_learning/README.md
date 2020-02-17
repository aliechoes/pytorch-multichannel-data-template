# Machine Learning	

Here you can find the machine learning part of the pipeline. It inlcudes the deap learning `models`, `losses`, `optimization` algorithms as well as the `metrics` which are going to be tracked.

## Models

At the moment, there is a simple deep learning model which can be simply used for test. It is called `ShallowNet`. Apart from that, `AlexNet` is implemented. The difference between a normal architecture with the implemented one is that the implemented ones can detect how many channels are available and initialized the model using that information. While this brings the limitation in case of pretrained models. It means that for models which have more or less than three channels, we cannot use pretrained models. However, it can benefit from the architecture.

In addition, every model includes a part called `embedding_generator` which can be used the extract features right away from the model with no need to tweak anything.
# Models

At the moment, there is a simple deep learning model which can be simply used for test. It is called `ShallowNet`. Apart from that, `AlexNet` and `ResNet18` are implemented. The difference between a normal architecture with the implemented one is that the implemented ones can detect how many channels are available and initialized the model using that information. While this brings the limitation in case of pretrained models. It means that for models which have more or less than three channels, we cannot use pretrained models. However, it can benefit from the architecture.

*Note*: In case there are three channels, the code automatically uses the pretrained weights from `torchvision`.

In addition, every model includes a part called `embedding_generator` which can be used to extract features right away from the model with no need to tweak anything.

For adding new models, you can simply add your model in the file [__init__.py](__init__.py). Please consider that it has to have the method `embedding_generator`. In addition, it should be added to the `config.json` to be considered.

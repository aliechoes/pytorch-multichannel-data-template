# pytorch-multichannel-data-template
In this repository, you can find pytorch based computer vision code which can be used for data with any number of channels. This is in general important as some applications such as satellite imagery or biomedical imaging can contain more than 3 channels of information.

.. TO BE UPDATED

## Libraries

```
Python 3.7.6 (Anaconda)
torch=='1.4.0'
tensorboard=='1.15.0'
pandas=='1.0.0'
sklearn=='0.22.1'
skimage=='0.16.2'
pillow=='6.2.0'
```

## Folders

* [configs](configs): includes the config files
* [inputs](inputs): includes the preprocessing models as well as the data loader
* [machine_learning](machine_learning): includes the models, metrics, losses as well as the optimizers.
* [outputs](outputs): includes the methods for tensorboard as well as saving results

## TODO

- [ ] add documentation
- [x] add tensorboard 
- [ ] add config tracker in tensorboard
- [ ] add pr-cruve
- [ ] add saving matplotlib saving using tensorboard
- [x] add more metrics 
- [x] add saving models
- [ ] add early stopping
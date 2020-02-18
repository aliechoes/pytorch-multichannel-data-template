# Pytorch Multichannel Image Classification

In this repository, you can find PyTorch based image classification code which can be used for data with any number of channels. This is in general important as some applications such as satellite imagery or biomedical imaging can contain more than 3 channels of information.

## Folder Structure

To Be Updated...

## Input parameters

For using this code, you need to use the config file following information:

 
* `data_dir` : directory of the images in their corresponding folders (each folder is a class) (format: str)
* `batch_size`:  Batch size
* `validation_split`: percentage of train-validation split
* `test_split`: percentage of test vs train-validation split
* `tensorboard_path`: path to save tensorboard as well as other outputs
* `file_extension`: file extension which exists in the data, for example `.png`
* `checkpoint`: the path, for transfer learning. In case it is not passed 
                            it will not be considered. The checkpoint should include: 
                `'epoch'`, `model.state_dict()`, `optimizer.state_dict()` and the `loss`.
* `model_name`: the used architecture. The name should be exactly the same from the 
                file [models.py](machine_learning/models.py):
* `num_epochs`: Number of epochs
* `device`: It can be `cpu` or `cuda`. I have intentionally kept this option and the
            code does not look if `cuda` exists
* `optimization_parameters`: a dictionary including the options which are needed for the [optimizers.py](machine_learning/optimizers.py)
* `optimization_method`: name of the optimizer in the file [optimizers.py](machine_learning/optimizers.py)
* `loss_function`: name of the loss function in the file [losses.py](machine_learning/losses.py)
* `metrics_of_interest`: list of metrics which should be trace. They should be same as the metrics in the files [metrics.py](machine_learning/metrics.py) 

## Running the Code

For running this code, it is enough to just pass the config file and run the [main.py](main.py):

```bash
python main.py --config ./configs/sample_config.json
```

.. TO BE UPDATED

## Libraries

This code is based on these libraries:

* `Python 3.7.6 (Anaconda)`
* `torch=='1.4.0'`
* `tensorboard=='1.15.0'`
* `pandas=='1.0.0'`
* `sklearn=='0.22.1'`
* `skimage=='0.16.2'`
* `pillow=='6.2.0'`


## Naming Convention

Each run is named using multiple conditions to be able to distinguish their results. The naming is based on:

```
DATETIME_NOW + _  + MODEL + _bs_ + BATCHSIZE
```

## Folders

* [configs](configs): includes the config files
* [inputs](inputs): includes the preprocessing models as well as the data loader
* [machine_learning](machine_learning): includes the models, metrics, losses as well as the optimizers.
* [outputs](outputs): includes the methods for tensorboard as well as saving results

## TODO

- [ ] add documentation
- [ ] add GRAD-CAM
- [ ] add tensorboard 
- [x] add tensorboard 
- [ ] add config tracker in tensorboard
- [ ] add pr-cruve
- [ ] add saving matplotlib saving using tensorboard
- [x] add more metrics 
- [x] add saving models
- [x] add transfer learning
- [ ] add data augmentation
- [ ] add early stopping
# Pytorch Multichannel Image Classification

In this repository, you can find PyTorch based image classification code which can be used for data with any number of channels. This is in general important as some applications such as satellite imagery or biomedical imaging can contain more than 3 channels of information.

## Folders

* [configs](configs): includes the config files
* [data_loaders](data_loaders): includes the dataset and dataloders
* [sample_data](sample_data): example of MNIST for running the code
* [preprocessing](preprocessing): includes the preprocessing methods
* [machine_learning](machine_learning): includes the models, metrics, losses as well as the optimizers.
* [tensorboard_writer](tensorboard_writer): includes the methods for tensorboard as well as saving results

## Input parameters

For using this code, you need to use the config file following information:
 
* `data_dir` : directory of the images in their corresponding folders (each folder is a class) (format: str)
* `batch_size`:  Batch size
* `validation_split`: percentage of train-validation split
* `data_map`: the map which can be used for mapping data from one range to another. In case `null` is passed, no transformation will be done on the data. At the moment, possible options are: `normalize`, `map_zero_one` and `map_minus_one_to_one`
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
* ``call_back``: dictionary of parameters which are needed for early stopping as well as saving models. It includes `saving_period` which indicates how many epochs are the models are saved. `patience` which the model can tolerate not getting improved as well as the `criteria` which we look at for best model selection. The `criteria` must be one of the `metrics_of_interest`. 


## Running the Code

For running this code, it is enough to just pass the config file and run the [main.py](main.py). For testing whether the code works or not, you can use the folder [sample_data](sample_data) which includes MNIST dataset with two channels (the second channel is artificially created)

```bash
python main.py --config ./configs/sample_config.json
```

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

## TODO

- [x] add documentation
- [x] update the dataloader to be able to get different file formats
- [ ] add options for augmentation from the config file
- [ ] add GRAD-CAM 
- [x] add tensorboard 
- [ ] add config tracker in tensorboard
- [x] add pr-cruve
- [ ] add saving matplotlib saving using tensorboard
- [x] add more metrics 
- [x] add saving models
- [x] add transfer learning
- [x] add data augmentation
- [x] add early stopping
- [x] add feature extration
- [x] add batch size for faster model evaluation
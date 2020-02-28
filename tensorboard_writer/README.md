# Tensorboard

 We use tensorboard for tracking the models and their metircs. 

 ## Currently existing parts

These are the currently parts for tensorboard:

- *scalars*: it tracks down all the metrics which are specified in the config file.
- *images*: it plots every channel per class per epoch
- *pr-curve*: it plots the precision-recall curve per class
- *graphs*: it plots the architecture
- *projector*: it plots the embeddings from the model for 200 random points per epoch

## Extra folders in the assigned folder for TB

In the folder which is specified for `tensorboard_path` from the config file, there are two extra folders as well

1. `models`: This is the place where the architectures are saved per their epoch as `.pth` files. In these file you can find `epoch`, `model_state_dict`, `optimizer_state_dict`, `loss`, `channels`, `statistics` and `data_map`.
2. `output_files`: this folder consisits of two files:
    - `aggregated_results.csv`: It includes the results per epoch for the train, validation and test sets.
    - `granular_results.csv`: It includes the results per sample for the last epoch. It is practically the dataframe `dataloader.df`.
# Sample Output

Here you can find the sample output folder as well as the loggings for running the code. In the [2020-02-28 22:10:40.241832_ShallowNet](folder) you can find the all the tensorboard outputs as well as the dataframes saved. In addition, you can find an example [sample_output.txt](sample_output).

*note*: the tensorboard files are omitted from the folder. However in a normal example it includes them

After training you can simply run the tensorboard from the main folder with:

```bash 
tensorboard --logdir <tensorboard_path used in the config file>
```
# Data Loaders

Here you can find the dataset and data loaders for Pytorch

## Dataset file structure

In the data folder, every folder is considered a separate class. There is no need to create train validation split as the `DataLoaderGenerator` does that automatically. In general it is expected that the `data_dir` has such structure (with the assumption we have 4 channels per image):

```
data_dir/class_x/xxx_Ch1.ext
data_dir/class_x/xxx_Ch2.ext
data_dir/class_x/xxx_Ch3.ext
data_dir/class_x/xxx_Ch4.ext
data_dir/class_x/xxy_Ch1.ext
data_dir/class_x/xxy_Ch2.ext
data_dir/class_x/xxy_Ch3.ext
data_dir/class_x/xxy_Ch4.ext
.
.
.
data_dir/class_y/123_Ch1.ext
data_dir/class_y/123_Ch2.ext
data_dir/class_y/123_Ch3.ext
data_dir/class_y/123_Ch4.ext
data_dir/class_y/h20_Ch1.ext
data_dir/class_y/h20_Ch2.ext
data_dir/class_y/h20_Ch3.ext
data_dir/class_y/h20_Ch4.ext
.
.
.
```
where each file is a single-channel image. That means that this code can also be used for three channels data, each image should be breakdown to three channels first. For example if we have a file names `my_image.jpg`, it should be broked down to `my_image_Ch1.jpg`, `my_image_Ch2.jpg` and `my_image_Ch3.jpg`.

## Possible Channels

At the moment, the code can handle files with `Ch1` to `Ch18`. it is expected that the files named like this:

- `SOMENAME_Ch1.ext`
- `SOMENAME_Ch2.ext`
- .
- .
- .
- `SOMENAME_Ch18.ext`

In addition, it does not matter whether the channels are starting from `Ch1`. The code can handle any random variation of the channels. It is in particular important as sometimes, we would like to keep track of the channels which were used in the experiment. 

## Image Extensions

The images should be one of the supporting formats by `imageio`
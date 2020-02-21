import numpy as np
from imageio import imread, imwrite
import glob
import os
from skimage import filters


def segment( file_path,  
            segmentatable_channels ,
            whole_object_segmentation_channel):
    
    image = imread(file_path)
    image = image.astype(np.uint16)

    for ch in segmentatable_channels:
        if ch in file_path:
            threshold = filters.threshold_yen(image)
            mask = (image > threshold).astype(np.uint16)
            image = image*mask
            image = image.astype(np.uint16)
            return image, mask 

    possible_channels = ["Ch1.","Ch2.","Ch3.","Ch4.","Ch5.","Ch6.","Ch7.",
                            "Ch8.","Ch9.","Ch10.","Ch11.","Ch12."]
    for ch in possible_channels:
        if ch in file_path:
            mask = imread(file_path.replace(ch,whole_object_segmentation_channel))
            threshold = filters.threshold_triangle(mask)
            mask = (mask > threshold).astype(np.uint16)
            image = image*mask
            image = image.astype(np.uint16)
            return image, mask 
    raise Exception('The channels seem inconsistent')

    
def add_border(image, desired_size):
    try:  
        assert (desired_size > image.shape[0]) & (desired_size > image.shape[1])
        
        if len(image.shape) == 2:
            desired_photo = np.ones((desired_size, desired_size ))*image[0,0]
            w = image.shape[1]
            h = image.shape[0]
            w0 = int((desired_size - w)/2)
            h0 = int((desired_size - h)/2)
            desired_photo[h0:h0+h,w0:w0+w] = image
        elif len(image.shape) == 3:
            desired_photo = np.ones((desired_size, desired_size, image.shape[2])) 
            w = image.shape[1]
            h = image.shape[0]
            w0 = int((desired_size - w)/2)
            h0 = int((desired_size - h)/2)

            for ch in range(image.shape[2]): 
                desired_photo[:,:,ch] = desired_photo[:,:,ch]*image[0,0,ch]
                desired_photo[h0:h0+h,w0:w0+w,ch] = image[:,:,ch]
            
        desired_photo = desired_photo.astype(np.uint16)
        return desired_photo, True
    except AssertionError:
        print("the file already bigger than the desired size")
        return image, False
    
  

def main(   images_path, 
            desired_size = 200, 
            segmentatable_channels = [],
            whole_object_segmentation_channel = "Ch4." ,
            existing_extension = ".tiff" ,
            desired_extension = ".png" ):
    x_min = desired_size
    x_max = 0
    list_of_files = glob.glob(os.path.join(images_path,"*" + existing_extension  ))
    
    for f in list_of_files:
        try:
            image = imread(f)
            image, mask = segment(f,  
                                segmentatable_channels ,
                                whole_object_segmentation_channel)
            image, to_save = add_border(image, desired_size)
            mask,  to_save = add_border(mask, desired_size)
            mask = mask.astype(np.uint8)*255
            x_min = min(    x_min, 
                            np.where(mask.sum(0) > 0)[0][0], 
                            np.where(mask.sum(1) > 0)[0][0])
            x_max = max(     x_max, 
                            np.where(mask.sum(0) > 0)[0][-1], 
                            np.where(mask.sum(1) > 0)[0][-1]) 
            if to_save:
                imwrite(f.replace(existing_extension,desired_extension ), 
                                                        image, 'PNG-PIL')
                imwrite(f.replace("images", "masks").replace(existing_extension, desired_extension ), 
                                                        mask ) 
        except ValueError as e: 
            print(image.shape)
            print(f)
            print(e)
    for f in list_of_files:
         os.remove(f)
    
    
    return x_min, x_max
 
if __name__ == '__main__':
    main_path = [
                "/pstore/home/shetabs1/data/aggregation_bodies/raw/experiment_7/images/7_a_host/" ,
                "/pstore/home/shetabs1/data/aggregation_bodies/raw/experiment_7/images/7_b_CD19cd28/" ,
                "/pstore/home/shetabs1/data/aggregation_bodies/raw/experiment_7/images/7_c_FAPcd137/" ] 
    desired_size = 200
    x_min = desired_size
    x_max = 0
    for mp in main_path:
        print(mp)
        x, y = main(mp, desired_size)
        x_min = min(x, x_min)
        x_max = max(y, x_max)
    print(x_min)
    print(desired_size- x_max)
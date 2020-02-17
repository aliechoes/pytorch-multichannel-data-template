import numpy as np
from imageio import imread, imwrite
import glob
import os

def main(main_path, desired_size = 256):
    list_of_files = glob.glob(os.path.join(main_path,"*.png"))
    for f in list_of_files:
        try:
            image = imread(f)
            image, to_save = add_border(image, desired_size)
            if to_save:
                imwrite(f, image)
        except ValueError as e: 
            print(image.shape)
            print(f)
            print(e)
    return None

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
        return desired_photo, True
    except AssertionError:
        #print("the file already bigger than the desired size")
        return image, False
    


if __name__ == '__main__':
    main_path = [
                "/pstore/data/DS4/aggregation_bodies/amnis_data/experiment_6/clone_90"]
    desired_size = 256
    for mp in main_path:
        print(mp)
        main(mp, desired_size)
import numpy as np
from imageio import imread, imwrite
import glob
import os
from skimage import filters



def main(   main_path,  
            fill_extension = ".jpg"):
    
    list_of_files = glob.glob(os.path.join(main_path, "*" + fill_extension)) 
    for f in list_of_files: 
        try:
             
            if "Ch4" in f:
                image = imread(f)
                image = image.astype(np.uint16)
                T = filters.threshold_li(image)
                image = (image > T).astype(int)
            
            elif "Ch5" in f:
                image = imread(f)
                image = image.astype(np.uint16)
                T = filters.threshold_yen(image)
                image = (image > T).astype(int)
            
            elif "Ch6" in f:
                image = imread(f)
                image = image.astype(np.uint16)
                T = filters.threshold_yen(image)
                image = (image > T).astype(int)
            else:
                os.remove(f)
                continue
            
            

            if fill_extension == ".jpg":
                imwrite(f , image ) 
                image = None

        except ValueError as e: 
            print(image.shape)
            print(f)
            print(e)
    return None
 

if __name__ == '__main__':
    main_path = [
                "/home/ali/Desktop/synapse_formation_test/Plus_SEA_Synapse/",
                "/home/ali/Desktop/synapse_formation_test/Plus_SEA_Synapse_Maybe/" ] 
    for mp in main_path:
        print(20*"-")
        print(mp)
        main(mp )
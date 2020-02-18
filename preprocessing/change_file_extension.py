import numpy as np
from imageio import imread, imwrite
import glob
import os

def main(   main_path, 
            existing_extension = ".jpg" ,
            desired_extension = ".jpg"):
    
    list_of_files = glob.glob(os.path.join(main_path, "*" + existing_extension))
    for f in list_of_files:
        try:
            image = imread(f)
            image = image.astype(np.uint16) 
            if desired_extension == ".jpg":
                imwrite(f.replace(existing_extension,desired_extension ), image, 'jpg-PIL')
                os.remove(f)
        except ValueError as e: 
            print(image.shape)
            print(f)
            print(e)
    return None
 

if __name__ == '__main__':
    main_path = [
                "/home/ali/Desktop/synapse_formation/Plus_SEA_Synapse",
                "/home/ali/Desktop/synapse_formation/Plus_SEA_Synapse_Maybe" ] 
    for mp in main_path:
        print(mp)
        main(mp )
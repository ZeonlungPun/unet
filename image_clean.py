import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

#get all the images path names
raw_images=glob("E:\\UNET\\IMAGES\\*.jpg")

#resize and save images
def save_images(images,save_dir,size=(768,512)):
    idx=1
    for path in tqdm(images,total=len(images)):
        x=cv2.imread(path,cv2.IMREAD_COLOR)
        x=cv2.resize(x,(size[1],size[0]))
        cv2.imwrite(f"{save_dir}/{idx}.jpg",x)
        idx+=1

save_images(raw_images,"E:\\UNET\\clean_images")
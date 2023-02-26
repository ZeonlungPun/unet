import os,cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip,CoarseDropout,RandomBrightness,RandomContrast

#dataset loading
def load_datasets(path):
    images=sorted(glob(os.path.join(path,"clean_images","*")))
    masks=sorted(glob(os.path.join(path,"mask","*")))
    return images,masks

dataset_path="E:\\UNET\\"
images,masks=load_datasets(dataset_path)

def save_datasets(images,masks,save_dir,augment=True):
    for x,y in tqdm(zip(images,masks),total=len(images)):
        name=x.split("\\")[-1].split(".")[0]
        x=cv2.imread(x,cv2.IMREAD_COLOR)
        y=cv2.imread(y,cv2.IMREAD_COLOR)


        if augment:
            aug=HorizontalFlip(p=1)
            augmented=aug(image=x,mask=y)
            x1=augmented["image"]
            y1=augmented["mask"]

            aug=CoarseDropout(p=1,max_holes=10,max_height=32,max_width=32)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug=RandomBrightness(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug=RandomContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            aug_x=[x,x1,x2,x3,x4]
            aug_y=[y,y1,y2,y3,y4]
        else:
            aug_x=[x]
            aug_y=[y]
        idx=0
        for ax,ay in zip(aug_x,aug_y):
            aug_name=f"{name}_{idx}.jpg"
            save_img_dir=os.path.join(save_dir,"images",aug_name)
            save_mask_dir=os.path.join(save_dir,"masks",aug_name)

            cv2.imwrite(save_img_dir,ax)
            cv2.imwrite(save_mask_dir,ay)
            idx+=1

aug_path="E:\\UNET\\aug"
save_datasets(images,masks,aug_path)
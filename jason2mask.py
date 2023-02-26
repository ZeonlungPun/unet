import json,cv2
import numpy as np

#read json file
f=open("mask.json","r")
data=json.load(f)

img_dir="E:\\UNET\\clean_images"
mask_dir="E:\\UNET\\mask"

images=data["images"]
annots=data["annotations"]

for x,y in zip(images,annots):
    filename=x['file_name']
    h=x['height']
    w=x['width']
    mask=np.zeros((h,w))
    seg=y['segmentation']
    for points in seg:
        contours=[]
        for i in range(0,len(points),2):
            contours.append((points[i],points[i+1]))
        contours=np.array(contours,dtype=np.int32)
        cv2.drawContours(mask,[contours],-1,255,-1)
    cv2.imwrite(f'{mask_dir}/{filename}',mask)
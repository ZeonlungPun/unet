import os,cv2
import numpy as np
from glob import glob
import tensorflow as tf
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Conv2DTranspose,Concatenate,Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,CSVLogger

#seeding
os.environ['PYTHONHASHSEED']=str(42)
np.random.seed(42)
tf.random.set_seed(42)

#hyperparameters
batch_size=2
lr=1e-4
epochs=100
height,width=768,512

#path
file_dir="E:\\UNET"
model_dir=os.path.join(file_dir,"unet.h5")
log_file=os.path.join(file_dir,"log.csv")
dataset_path="E:\\UNET\\data"

#UNET
#CONV BLOCK
def conv_block(inputs,num_filters):
    x=Conv2D(num_filters,3,padding="same")(inputs)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x
#encoder
def encoder_block(inputs,filters):
    x=conv_block(inputs,filters)
    p=MaxPool2D((2,2))(x)
    return x,p
#decoder
def decoder_block(inputs,skip,num_filters):
    x=Conv2DTranspose(num_filters,(2,2),strides=2,padding="same")(inputs)
    x=Concatenate()([x,skip])
    x=conv_block(x,num_filters)
    return x

def build_unet(input_shape):
    inputs=Input(input_shape)
    s1,p1=encoder_block(inputs,64)
    s2,p2=encoder_block(p1,128)
    s3,p3=encoder_block(p2,256)
    s4,p4=encoder_block(p3,512)
    b1=conv_block(p4,1024)
    d1=decoder_block(b1,s4,512)
    d2=decoder_block(d1,s3,256)
    d3=decoder_block(d2,s2,128)
    d4=decoder_block(d3,s1,63)

    outputs=Conv2D(1,1,padding="same",activation="sigmoid")(d4)
    model=Model(inputs,outputs,name="unet")
    return model

#data pipeline
def load_data(path):
    train_x=sorted(glob(os.path.join(path,"train","clean_images","*")))
    train_y = sorted(glob(os.path.join(path, "train", "mask", "*")))

    val_x = sorted(glob(os.path.join(path, "val", "clean_images", "*")))
    val_y = sorted(glob(os.path.join(path, "val", "mask", "*")))
    return (train_x,train_y),(val_x,val_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    return x

def read_mask(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x=x/255.0
    x=np.expand_dims(x,axis=-1)
    return x
#tf.data.pipeline
def tf_parse(x,y):
    def _parse(x,y):
        x=read_image(x)
        y=read_mask(y)
        return x,y
    x,y=tf.numpy_function(_parse,[x,y],[tf.float64,tf.float64])
    x.set_shape([height,width,3])
    y.set_shape([height, width, 1])
    return x,y
def tf_dataset(x,y,batch=4):
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset=dataset.map(tf_parse,num_parallel_calls=tf.data.AUTOTUNE)
    dataset=dataset.batch(batch)
    dataset=dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

#training
(train_x,train_y),(val_x,val_y)=load_data(dataset_path)
print(f'train:{len(train_x)}-{len(train_y)}')

train_dataset=tf_dataset(train_x,train_y,batch=batch_size)
val_dataset=tf_dataset(val_x,val_y,batch=batch_size)

input_shape=(height,width,3)
model=build_unet(input_shape)

opt=tf.keras.optimizers.Adam(lr)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["acc"])
callbacks=[ModelCheckpoint(model_dir,verbose=1,save_best_only=True),
           ReduceLROnPlateau(monitor="val_loss",factor=0.1,patience=4),
           CSVLogger(log_file),
           EarlyStopping(monitor="val_loss",patience=20,restore_best_weights=False)]
model.fit(train_dataset,validation_data=val_dataset,epochs=epochs,callbacks=callbacks)
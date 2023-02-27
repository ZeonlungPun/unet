import os,cv2
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary


#set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#data pipeline
def read_image(path):
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0
    return x

def read_mask(path):
    y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    y=y/255.0
    y=np.expand_dims(y,axis=-1)
    return y


class UnetDataset(Dataset):
    def __init__(self,dataset_path):
        super(UnetDataset, self).__init__()
        self.dataset_path=dataset_path
        self.img_path=self.dataset_path+"\\images"
        self.mask_path=self.dataset_path+"\\masks"
        self.img_names=sorted(glob(os.path.join(self.img_path,"*")))
        self.mask_names=sorted(glob(os.path.join(self.mask_path,"*")))


    def __getitem__(self, index):
        img_name,mask_name=self.img_names[index],self.mask_names[index]
        img,mask=read_image(img_name),read_mask(mask_name)

        return img,mask

    def __len__(self):
        return len(self.mask_names)

batch_size=2

train_data=UnetDataset("E:\\UNET\\aug\\train")
train_loader=DataLoader(train_data,batch_size=batch_size,shuffle=True)
val_data=UnetDataset("E:\\UNET\\aug\\val")
val_loader=DataLoader(val_data,batch_size=batch_size,shuffle=False)


#model
#CONV BLOCK : size of img not change; channels enlarge
class CONV_BLOCK(nn.Module):
    def __init__(self,in_channels,num_filters):
        super(CONV_BLOCK, self).__init__()
        self.conv1=nn.Conv2d(in_channels,num_filters,kernel_size=(3,3),padding="same")
        self.conv2=nn.Conv2d(num_filters,num_filters,kernel_size=(3,3),padding="same")
        self.BN=nn.BatchNorm2d(num_filters)

    def forward(self,x):
        x=self.conv1(x)
        x=F.relu(self.BN(x))

        x=self.conv2(x)
        x = F.relu(self.BN(x))
        return x

class Encoder_block(nn.Module):
    def __init__(self,inputs,filters):
        super(Encoder_block, self).__init__()
        self.conv=CONV_BLOCK(inputs,filters)
        self.maxpool2d=nn.MaxPool2d(kernel_size=(2,2))

    def forward(self,x):
        x=self.conv(x)
        p=self.maxpool2d(x)
        return x,p

class Decoder_block(nn.Module):
    def __init__(self,inputs,num_filters):
        super(Decoder_block, self).__init__()
        #self.convTran=nn.Upsample(scale_factor=2,mode="nearest")
        self.convTran=nn.ConvTranspose2d(inputs,num_filters,kernel_size=(2,2),stride=(2,2))
        self.conv=CONV_BLOCK(inputs,num_filters)
    def forward(self,x,skip):
        # (H,W,C)--> (H*2,W*2,num_filters)
        x=self.convTran(x)
        # (H*2,W*2,num_filters) --> ( H*2,W*2, 2*num_filter(inputs) )
        x=torch.cat([x,skip],dim=1)
        # ( H*2,W*2, 2*num_filter(inputs) ) --> ( H*2,W*2, num_filters )
        x=self.conv(x)
        return x



#check the models output
"""
model=Decoder_block(1024,512).to(device)
s=summary(model,input_size=(1024,48,32))
print(s)
model1=CONV_BLOCK(3,64)
model1.to(device)
s1=summary(model1,input_size=(3,768, 512))

model2=Encoder_block(3,64).to(device)
s2=summary(model2,input_size=(3,768, 512))
print(s2)

model3=Encoder_block(64,128).to(device)
s3=summary(model3,input_size=(64, 384, 256))
print(s3)

model4=Encoder_block(128,256).to(device)
s4=summary(model4,input_size=(128,192, 128))
print(s4)
"""
class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.encoder1=Encoder_block(3,64)
        self.encoder2=Encoder_block(64,128)
        self.encoder3=Encoder_block(128,256)
        self.encoder4=Encoder_block(256,512)
        self.conv=CONV_BLOCK(512,1024)
        self.decoder1 = Decoder_block(1024, 512)
        self.decoder2 = Decoder_block(512,  256)
        self.decoder3 = Decoder_block(256,  128)
        self.decoder4 = Decoder_block(128,  64)
        self.conv_last=nn.Conv2d(64,1,kernel_size=(1,1),padding="same")


    def forward(self,x):
        s1,p1=self.encoder1(x)
        s2,p2=self.encoder2(p1)
        s3,p3=self.encoder3(p2)
        s4,p4=self.encoder4(p3)
        b1=self.conv(p4)
        d1 = self.decoder1(b1,s4)
        d2 = self.decoder2(d1,s3)
        d3 = self.decoder3(d2,s2)
        d4 = self.decoder4(d3,s1)
        out=F.sigmoid(self.conv_last(d4))

        return out


def fit_one_epoch(epoch,epochs,model,train_loader,val_loader):
    loss_fun=torch.nn.BCELoss()
    optim=torch.optim.SGD(model.parameters(),lr=0.01)

    correct = 0
    total = 0
    sum_loss = 0

    #train mood
    model.train()
    loop=tqdm(train_loader,desc="training")
    for x,y in loop:
        x,y=x.to(device),y.to(device)
        x,y= x.transpose(1,3).transpose(2,3).float(),y.transpose(1,3).transpose(2,3).float()
        y_pred=model(x)

        loss=loss_fun(y_pred,y)
        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            y_pred=(y_pred>0.5)
            yy=y.reshape((-1,1))
            yypred=y_pred.reshape((-1,1))
            total+=yy.shape[0]
            correct+= (yy==yypred).sum().item()
            running_acc = correct / total
            sum_loss += loss.item()
            running_loss = sum_loss /total
            # update training info
            loop.set_description(f'Epoch [{epoch + 1}/{epochs}]')
            loop.set_postfix(loss=running_loss, acc=running_acc)

    epoch_loss = sum_loss / total
    epoch_acc = correct / total

    #test mode
    test_correct = 0
    test_total = 0
    test_sum_loss = 0

    model.eval()
    with torch.no_grad():
        loop2 = tqdm(val_loader, desc='Test')
        for x, y in loop2:
            x, y = x.to(device), y.to(device)
            x, y = x.transpose(1, 3).transpose(2, 3).float(), y.transpose(1, 3).transpose(2, 3).float()
            y_pred = model(x)
            loss = loss_fun(y_pred, y)
            y_pred=(y_pred>0.5)
            yy = y.reshape((-1, 1))
            yypred = y_pred.reshape((-1, 1))
            test_correct += (yypred == yy).sum().item()
            test_total += yy.shape[0]
            test_sum_loss += loss.item()
            test_running_loss = test_sum_loss / test_total
            test_running_acc = test_correct / test_total

            loop2.set_postfix(loss=test_running_loss, acc=test_running_acc)
        test_epoch_loss = test_sum_loss / test_total
        test_epoch_acc = test_correct / test_total

    return epoch_loss, epoch_acc, test_epoch_loss, test_epoch_acc

def train(epochs,model, train_loader, val_loader,saving_way):
    train_loss = []
    train_acc  = []
    val_loss = []
    val_acc  = []
    max_val_acc=0
    for epoch in range(epochs):
        epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc = fit_one_epoch(epoch, epochs,model, train_loader, val_loader)
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)
        val_loss.append(val_epoch_loss)
        val_acc.append(val_epoch_acc)
        max_val_acc=max(val_acc)
        print("save model")
        if saving_way == "parameters":
            torch.save(model.state_dict(), "unet_model_epoch{}.pth".format(epoch))
        elif saving_way == "all":
            torch.save(model, "unet_model_epoch{}.pth".format(epoch))


        if val_epoch_acc >= max_val_acc:
            print("save best model")
            if saving_way=="parameters":
                torch.save(model.state_dict(),"best_unet_model.pth")
            elif saving_way=="all":
                torch.save(model,"best_unet_model.pth")

def UnetPredict(img_path,saving_way,model_path):
    #load model
    if saving_way == "parameters":
        model=Unet().to(device)
        model.load_state_dict(torch.load(model_path))
    else:
        model=torch.load(model_path).to(device)
    model.eval()
    with torch.no_grad():
        #load img
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)/255.0
        img=cv2.resize(img,( 512,768))
        img=torch.tensor(np.expand_dims(img,0))
        img=img.to(device).transpose(1,3).transpose(2,3).float()
        pred_mask=model(img)
        pred_mask=np.squeeze(pred_mask.cpu().transpose(1,2).transpose(2,3).numpy(),axis=0)
        pred_mask=np.round(pred_mask)
        return pred_mask

if __name__=='__main__':
    #model = Unet().to(device)
    #s = summary(model, input_size=(3, 768, 512))
    #print(s)
    #train(6,model,train_loader, val_loader,saving_way="all")
    predict_mask=UnetPredict("E:\\UNET\\aug\\train\\images\\3_1.jpg","all","best_unet_model.pth")
    print(predict_mask)
    cv2.imshow("mask",predict_mask)
    cv2.waitKey(0)










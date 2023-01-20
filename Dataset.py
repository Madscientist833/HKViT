import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
import pandas as pd
def get_float_bin_data3D(filename, nx, ny, nz):
    pic = np.fromfile(filename,dtype=np.float32)
    pic = pic.reshape(nx,ny,nz)
    return pic

def get_float_bin_data2D(filename, nx, ny):
    pic = np.fromfile(filename,dtype=np.float32)
    pic = pic.reshape(nx,ny)
    return pic
# get H and K from the distribution
def get_HK_from_label(label):
    label_shape,_=label.shape
    H_label=np.zeros(label_shape)
    K_label=np.zeros(label_shape)
    for idx in range(label_shape):
        label1=label[idx,:]
        
        hk=np.reshape(label1,1050)
        h=hk[0:650]
        yy=0
        vv=0
        for i in range(650):
            if yy <= h[i]:
                yy=h[i]
                vv=i
        hh=(vv-100)*100+30000 
        H_label[idx]=hh
        
        k=hk[650:1050]
        yy=0
        vv=0
        for i in range(0, 400, 1):
            if yy <= k[i]:
                yy=k[i]
                vv=i
        kk=(vv-100)*0.001+1.6
        K_label[idx]=kk
    return H_label,K_label
# get dataloader for train and test
def get_loader(is_train=True, is_original=False, transform=None, Root='./data/',LabelType='H',Batchsize=4):
    
    test_label_path=Root+'Test_Hk10000_1050.dat'
    train_label_path=Root+'Train_Hk60000_1050.dat'
    
    if(is_original):
        test_data_path=Root+'Orig_test_RF10000_73_500.dat'
        train_data_path=Root+'Orig_train_RF60000_73_500-004.dat'
    else:
        test_data_path=Root+'Test_RF10000_73_500.dat'
        train_data_path=Root+'Train_RF60000_73_500-003.dat'
    
    if(is_train):
        train_set=RFdataset(train_data_path,train_label_path,60000,LabelType,transform)
        train_loader = DataLoader(train_set,
                                  batch_size=Batchsize,
                                  shuffle=True,
                                  num_workers=4)
        return train_loader
    else:
        train_set=RFdataset(test_data_path,test_label_path,10000,LabelType,transform)
        train_loader = DataLoader(train_set,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=4)
        return train_loader
# RF dataset for regression task
class RFdataset(Dataset):
    def __init__(self, RF_file, HK_file, dataset_numbers, LabelType="H",transform=None):
        self.RF_file=RF_file
        self.HK_file=HK_file
        self.dataset_numbers=dataset_numbers
        self.LabelType=LabelType
        self.transform = transform
        if(self.transform is None):
            train_transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.transform=train_transform
        
        print("Loading Data...")
        self.RF_data= get_float_bin_data3D(RF_file,dataset_numbers,73,500)
        self.label= get_float_bin_data2D(HK_file,dataset_numbers,1050)
        print("Data Loaded!\n")
        
        print("Processing Data...")
        self.H_label,self.K_label=get_HK_from_label(self.label)
        self.H_label=self.H_label/1000
        self.H_label=torch.tensor(self.H_label)
        self.K_label=torch.tensor(self.K_label)
        self.H_label=torch.reshape(self.H_label,(dataset_numbers,1))
        self.K_label=torch.reshape(self.K_label,(dataset_numbers,1))
        print("Data Processed!")
    def __getitem__(self, idx):
        img=self.RF_data[idx]
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = self.transform(img)
        if(self.LabelType=='H' or self.LabelType=='h'):
            Hlabel=self.H_label[idx]
            return img, Hlabel
        elif(self.LabelType=='K' or self.LabelType=='k'):
            Klabel=self.K_label[idx]
            return img, Klabel
        else:
            Hlabel=self.H_label[idx]
            Klabel=self.K_label[idx]
            return img, Hlabel, Klabel
    def __len__(self):
        return len(self.H_label)
# Real dataset for regression task
class Realdataset(Dataset):
    def __init__(self, RF_root, label_file, LabelType="H",transform=None,Denoise=None):
        self.label=pd.read_table(label_file)
        self.label['data']=self.label['STATION'].map(lambda x:RF_root+x+'.only_r.rf')
        self.LabelType=LabelType
        self.transform = transform
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if(self.transform is None):
            train_transform=transforms.Compose([transforms.Resize((224, 224), Image.BILINEAR),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            self.transform=train_transform
        if(Denoise is not None):
            self.Denoise_net=torch.load(Denoise)
        else:
            self.Denoise_net=None
    def __getitem__(self, idx):
        img=get_float_bin_data2D(self.label.loc[idx,'data'],73,500)
        if(self.Denoise_net is not None):
            img=torch.tensor(img.reshape(1,1,73,500))
            img=img.to(self.device)
            img=self.Denoise_net(img).cpu().detach().numpy().reshape(73,500)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = Image.fromarray(img, mode='RGB')
        img = self.transform(img)
        if(self.LabelType=='H' or self.LabelType=='h'):
            Hlabel=self.label.loc[idx,'H (km)']
            return img, Hlabel
        elif(self.LabelType=='K' or self.LabelType=='k'):
            Klabel=self.label.loc[idx,'k']
            return img, Klabel
        else:
            Hlabel=self.label.loc[idx,'H (km)']
            Klabel=self.label.loc[idx,'k']
            return img, Hlabel, Klabel
    def __len__(self):
        return self.label.shape[0]
# Denoise dataset
class Denoisedataset(Dataset):
    def __init__(self, RF_root='/home/featurize/data/data/', dataset_numbers=60000,is_train=True):
        self.RF_root=RF_root
        self.dataset_numbers=dataset_numbers
        self.is_train=is_train
        print("Loading Data...")
        if(is_train):
            self.RF_data=get_float_bin_data3D(self.RF_root+'Train_RF60000_73_500-003.dat',dataset_numbers,73,500)
            self.Ori_data=get_float_bin_data3D(self.RF_root+'Orig_train_RF60000_73_500-004.dat',dataset_numbers,73,500)
        else:
            self.RF_data=get_float_bin_data3D(self.RF_root+'Test_RF10000_73_500.dat',dataset_numbers,73,500)
            self.Ori_data=get_float_bin_data3D(self.RF_root+'Orig_test_RF10000_73_500.dat',dataset_numbers,73,500)
        self.RF_data=self.RF_data.reshape(dataset_numbers,1,73,500)
        self.Ori_data=self.Ori_data.reshape(dataset_numbers,1,73,500)
        print("Data Loaded!\n")
    
    def __getitem__(self, idx):
        img=self.RF_data[idx]
        target=self.Ori_data[idx]
        img=torch.tensor(img)
        target=torch.tensor(target)
        return img,target
    
    def __len__(self):
        return self.dataset_numbers

        
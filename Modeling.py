import numpy as np
import torch
import torch.nn as nn
import torchvision.models
from vit_pytorch import SimpleViT,deepvit,cait
from timm import create_model as creat

#get defined CNN for regression task
def get_pretrained_CNN(model_type="resnet18",out_channel=1):
    if(model_type=="resnet18"):
        mymodel=torchvision.models.resnet18()
        mymodel.fc=nn.Linear(512, out_channel)
    elif(model_type=='resnet50'):
        mymodel=torchvision.models.resnet50()
        mymodel.fc=nn.Linear(2048, out_channel)
    elif(model_type=='alexnet'):
        mymodel=torchvision.models.alexnet()
        mymodel.classifier[6]=nn.Linear(4096, out_channel)
    elif(model_type=='vgg16'):
        mymodel=torchvision.models.vgg16()
        mymodel.classifier[6]=nn.Linear(4096, out_channel)
    elif(model_type=='vgg19'):
        mymodel=torchvision.models.vgg19()
        mymodel.classifier[6]=nn.Linear(4096, out_channel)
    else:
        print("unsupported model type!")
        return
    return mymodel
#get ViT for regression task
def get_vit(model_type="SimpleViT",out_channel=1):
    if(model_type=='SimpleViT'):
        mymodel=SimpleViT(
            image_size = 224,
            patch_size = 32,
            num_classes = out_channel,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )
    elif(model_type=='DeepViT'):
        mymodel=deepvit.DeepViT(
            image_size = 224,
            patch_size = 32,
            num_classes = out_channel,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048
        )
    elif(model_type=='CaiT'):
        mymodel=cait.CaiT(
            image_size = 224,
            patch_size = 32,
            num_classes = out_channel,
            dim = 1024,
            depth = 12,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
    else:
        print("unsupported model type!")
        return
    return mymodel
#get pretrained ViT for regression task
def get_vit_pretrained():
    model = creat('vit_base_patch16_224', pretrained=True, num_classes=1)
    return model

#simple CNN for denoise
class DenoiseNet(nn.Module):
    def __init__(self,):
        super(DenoiseNet, self).__init__()
        self.conv1=nn.Conv2d(1,5,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(5,5,kernel_size=3,padding=1)
        self.conv22=nn.Conv2d(5,5,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(5,1,kernel_size=3,padding=1)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.relu(x)
        x=self.conv22(x)
        x=self.relu(x)
        x=self.conv3(x)
        return x





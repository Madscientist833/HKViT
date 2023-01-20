import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm

from Dataset import get_loader,Denoisedataset
from Modeling import get_pretrained_CNN,get_vit,get_vit_pretrained,DenoiseNet
from Scheduler import WarmupConstantSchedule,WarmupCosineSchedule,WarmupLinearSchedule


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
myLoss = torch.nn.MSELoss()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger
# training regression model
def train_reg(lr=1e-3,reg_target='H',modelType='resnet18',is_original=False,batch_size=4,maxepoch=5):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    myLoss = torch.nn.MSELoss()
    if modelType in ['resnet18','resnet50','alexnet','vgg16','vgg19']:
        mymodel=get_pretrained_CNN(modelType)
    elif modelType in ['SimpleViT','DeepViT','CaiT']:
        mymodel=get_vit(modelType)
    elif modelType=='pre':
        mymodel=get_vit_pretrained()
    else:
        print("invalid model type！")
        return
    myModel = mymodel.to(device)
    myOptimzier = optim.SGD(myModel.parameters(), lr = lr, momentum=0.9)
    
    train_loader=get_loader(is_train=True,
                            is_original=is_original, 
                            transform=None, 
                            Root='/home/featurize/data/data/',
                            LabelType=reg_target,
                            Batchsize=batch_size)
    test_loader =get_loader(is_train=False,
                            is_original=is_original, 
                            transform=None, 
                            Root='/home/featurize/data/data/',
                            LabelType=reg_target,
                            Batchsize=1)
    if(is_original):issyn='ori'
    else:issyn='syn'
    logger = get_logger('./log/'+modelType+'-'+reg_target+'-'+issyn+'.log')
    
    for _epoch in range(maxepoch):
        training_loss = 0.0
        loop= tqdm(enumerate(train_loader),total=len(train_loader))
        for _step, (image,label) in loop:
            image, label = image.to(device), label.to(device).float()
            predict_label = myModel.forward(image).float()
            loss = myLoss(predict_label, label)
            myOptimzier.zero_grad()
            loss.backward()
            myOptimzier.step()

            training_loss = training_loss + loss.cpu().detach().numpy().item()
            if _step % 1000 == 0:
                logger.info('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/1000))
                training_loss = 0.0
                logger.info('//')
        valoss = 0
        torch.save(myModel, './model/'+modelType+'-'+reg_target+'-'+issyn+'.pkl') # 保存整个模型
        myModel.eval()
        for step,(images,labels) in enumerate(test_loader):
            # GPU加速
            images = images.to(device)
            labels = labels.to(device).float()
            with torch.no_grad():
                outputs = myModel(images).float()
                tmploss = myLoss(labels,outputs)
                valoss += tmploss.cpu().detach().numpy().item()
        loop.set_description(f'Epoch [{_epoch}/{maxepoch}]')
        loop.set_postfix(loss = valoss/len(test_loader))

        logger.info('Validloss : %.3f' % ( valoss / len(test_loader)))
    return myModel
# training denoise network
def train_denoise(lr=1e-3,batch_size=4,maxepoch=5,modelname='./model/Denoise_net.pkl'):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    myLoss = torch.nn.MSELoss()
    mymodel=DenoiseNet()
    myModel = mymodel.to(device)
    myOptimzier = optim.SGD(myModel.parameters(), lr = lr, momentum=0.9)
    train_set=Denoisedataset()
    test_set=Denoisedataset(is_train=False,dataset_numbers=10000)
    scheduler = WarmupLinearSchedule(myOptimzier, warmup_steps=500, t_total=(maxepoch*60000/batch_size))
    train_loader=DataLoader(train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4)
    test_loader= DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    
    logger = get_logger('./log/Denoise.log')
    for _epoch in range(maxepoch):
        training_loss = 0.0
        loop= tqdm(enumerate(train_loader),total=len(train_loader))
        for _step, (image,label) in loop:
            image, label = image.to(device), label.to(device).float()
            predict_label = myModel.forward(image).float()
            loss = myLoss(predict_label, label)
            myOptimzier.zero_grad()
            loss.backward()
            myOptimzier.step()
            scheduler.step()

            training_loss = training_loss + loss.cpu().detach().numpy().item()
            if _step % 1000 == 0:
                logger.info('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/1000))
                training_loss = 0.0
                logger.info('//')
        valoss = 0
        torch.save(myModel, modelname) # 保存整个模型
        myModel.eval()
        for step,(images,labels) in enumerate(test_loader):
            # GPU加速
            images = images.to(device)
            labels = labels.to(device).float()
            with torch.no_grad():
                outputs = myModel(images).float()
                tmploss = myLoss(labels,outputs)
                valoss += tmploss.cpu().detach().numpy().item()
        loop.set_description(f'Epoch [{_epoch}/{maxepoch}]')
        loop.set_postfix(loss = valoss/len(test_loader))

        logger.info('Validloss : %.3f' % ( valoss / len(test_loader)))
    return myModel
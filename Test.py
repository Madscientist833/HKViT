import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm


from Dataset import Realdataset, get_loader, Denoisedataset
# calculate PSNR
def torchPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps
# helper function for test
def Go_test(model,data_loader,device,myloss):
    train_loss=0
    sum_dk=0
    for step,(images,labels) in tqdm(enumerate(data_loader),total =len(data_loader)):
        images = images.to(device)
        labels = labels.to(device).float()
        with torch.no_grad():
            outputs = model(images).float()
            tmploss = myloss(labels,outputs)
            train_loss += tmploss.cpu().detach().numpy().item()
            dk=(abs(outputs-labels)/labels).sum().cpu().detach().numpy().item()
            sum_dk+=dk
    return train_loss/len(data_loader),sum_dk/len(data_loader)
# helper function for denoise_test
def Go_Denoise_test(model,data_loader,device,myloss):
    train_loss=0
    sum_PSNR=0
    for step,(images,labels) in tqdm(enumerate(data_loader),total =len(data_loader)):
        images = images.to(device)
        labels = labels.to(device).float()
        with torch.no_grad():
            outputs = model(images).float()
            tmploss = myloss(labels,outputs)
            train_loss += tmploss.cpu().detach().numpy().item()
            tmpPSNR=torchPSNR(labels,outputs)
            sum_PSNR+=tmpPSNR
    return train_loss/len(data_loader),sum_PSNR/len(data_loader)
# test model for regression task
def test(modelpath,LabelType='K',Denoise=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=torch.load(modelpath)
    myLoss = torch.nn.MSELoss()
    if(Denoise is None):
        is_original=False
    else:
        is_original=True
    train_loader=get_loader(is_train=True,
                            is_original=is_original, 
                            transform=None, 
                            Root='/home/featurize/data/data/',
                            LabelType=LabelType,
                            Batchsize=1)
    test_loader= get_loader(is_train=False,
                            is_original=is_original, 
                            transform=None, 
                            Root='/home/featurize/data/data/',
                            LabelType=LabelType,
                            Batchsize=1)
    real_data=Realdataset('/home/featurize/data/data/real_data/','Li_H_k.txt',LabelType,None,Denoise)
    real_loader= DataLoader(real_data,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)
    
    train_loss,train_dk= Go_test(model,train_loader,device,myLoss)
    print("MSE on Train set={},d{} on Train set={}".format(train_loss,LabelType,train_dk))
    test_loss,test_dk= Go_test(model,test_loader,device,myLoss)
    print("MSE on Test set={},d{} on Test set={}".format(test_loss,LabelType,test_dk))
    real_loss,real_dk= Go_test(model,real_loader,device,myLoss)
    print("MSE on Real set={},d{} on Real set={}".format(real_loss,LabelType,real_dk))
# test model for denoise task
def Denoise_test(modelpath):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model=torch.load(modelpath)
    myLoss = torch.nn.MSELoss()
    train_set=Denoisedataset()
    test_set=Denoisedataset(is_train=False,dataset_numbers=10000)
    
    train_loader=DataLoader(train_set,
                            batch_size=1,
                            shuffle=True,
                            num_workers=4)
    test_loader= DataLoader(test_set,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)
    
    train_loss,train_PSNR= Go_Denoise_test(model,train_loader,device,myLoss)
    print("MSE on Train set={},PSNR on Train set={}".format(train_loss,train_PSNR))
    test_loss,test_PSNR= Go_Denoise_test(model,test_loader,device,myLoss)
    print("MSE on Test set={},PSNR on Test set={}".format(test_loss,test_PSNR))
    
    

        
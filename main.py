from Test import test,Denoise_test
from Train import train_denoise,train_reg

train_denoise()
train_reg(is_original=True)
Denoise_test('./model/Denoise_net.pkl')
test('./model/resnet18_H','H','./model/Denoise_net.pkl')
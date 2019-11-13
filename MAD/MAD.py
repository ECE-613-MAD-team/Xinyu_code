import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms


import pytorch_ssim
import numpy as np
import copy
import cv2
import gc

from models import *
from opt import *

import inspect
from Pytorch_Memory_Utils import *

import warnings
warnings.filterwarnings("ignore")
"""
pebbles.jpg
brick_wall.jpg
lacelike.jpg
radish.jpg
red-peppers.jpg

einstein.png

"""




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256



cu = 0
iterations = 5000
lamda = 0.02    

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

theta_0 = 0
m_t = 0 
v_t = 0 

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image








def cv_converter(img):
    image = Image.fromarray(img[...,::-1])
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)    
 
    
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)




def imshow(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('pebbles_noise7_1.jpg',dpi = 300)
    plt.show()

    
def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('pebbles_noise7_3.jpg',dpi = 300)
    plt.show()
# plt.figure()
# imshow(ref_img, title='reference texture')

ref_img = image_loader("./data/texture/pebbles.jpg")
_, nc, imsize,_ = ref_img.shape

"""

gaussian blur


"""

# imgn = cv2.imread("./data/texture/pebbles.jpg")
# imgn = cv2.GaussianBlur(imgn, ksize = (0, 0),sigmaX = (6.5))


# imgn = cv_converter(imgn)

"""

gaussian noise


"""
#m = 128
#ref_img = ref_img[:,:,32:32+m,32:32+m]
#print(ref_img.shape)
k = 7
ref = ref_img * 255
#noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
imgn = (ref+noise) / 255
imgn = torch.clamp(imgn,0,1)



# plt.figure()
# imshow1(imgn, title='distorted image')
# plt.show()



model_style, style_losses = get_style_model_and_losses(cnn,
          cnn_normalization_mean, cnn_normalization_std, ref_img, device = device)

#print(model_style)

imgn.data.clamp_(0,1)
input_img = imgn.detach()
ref = ref_img.detach()

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame) 
for i in range(iterations):
#    if i%50 == 0:
#        lamda = lamda*0.9

#    if loss2 < 30000:
#        lamda = lamda*0.998
#    else:
#        lamda = lamda*0.940
    
    
    #loss1, g1 = model_gram(model_style, input_img.detach(), style_losses)
    
    gpu_tracker.track()
    loss1, g1 = mse(input_img.detach(), ref.detach())
    
    #loss1, g1 =ssim(input_img.detach(), ref.detach())
    
    if i%20 == 0:
        print('loss1',loss1)
        #plt.hist(g1.cpu(),1000)
        #plt.show()
        print('g1',g1.max(),g1.min(),torch.mean(torch.abs(g1)))
    

    #loss2, g2 = mse(input_img.detach(), ref.detach())
    gpu_tracker.track()
    loss2, g2 = model_gram(input_img.detach(), ref.detach(), device)
#    if loss2 > 10:
#        lamda = lamda*0.998
#    else:
#        lamda = lamda*0.940
   
    #loss2, g2 = ssim(input_img.detach(), ref.detach())
    
    
    if i%20 == 0:
        print('\n\n\n')
    
        print('loss2',loss2)
        #plt.hist(g2.cpu(),1000)
        #plt.show()
        print('g2',g2.max(),g2.min(),torch.mean(torch.abs(g2)))
#    if i > 0 :
#        sgn = torch.dot(g2,g_prev)
#        if sgn > 0:
#            lamda = 1.2*lamda
#        else:
#            lamda = lamda*0.2
    
    
    if i%20 == 0:
        print('lamda:', lamda)
        
#    m_t = beta_1*m_t + (1-beta_1)*g2     # consider 90% of previous, and 10% of current
#    v_t = beta_2*v_t + (1-beta_2)*(g2*g2) # 99.9% of previous (square grad), and 1% of current
#    m_cap = m_t/(1-(beta_1**(i+1)))      #calculates the bias-corrected estimates
#    v_cap = v_t/(1-(beta_2**(i+1)))
#    gt = (m_cap)/(torch.sqrt(v_cap)+epsilon)
    gpu_tracker.track()
    y, comp = search_grad(ref.detach(), g = g2, gkeep = g1, img = input_img.detach(), mkeep = mse, lamda = lamda)
    
    cu = cu + comp
    
    if i %20 == 0:
        print('cumulate comp:', cu)
        plt.figure()
        imshow(torch.clamp(y,0,1))
    if comp > 1:
        torch.save(y,'temp.pt')  
        break
    
    
        
    input_img = y
    
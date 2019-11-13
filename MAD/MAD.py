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
iterations = 10000
#lamda = 0.04    

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




def step_size(lamda0,opt,rate1,rate2,iteration):
    

    if iteration < opt:
        lamda = lamda0*(rate1**iteration)
    else:
        lamda = lamda0*(rate1**iteration)*(rate2**(iteration-opt))
            
    return lamda
            
        



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
    plt.savefig('pebbles_noise8_5.jpg',dpi = 300)
    plt.show()

    
def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('pebbles_noise6_3.jpg',dpi = 300)
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
k = 8
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

prev_loss = 0
count = 0
#frame = inspect.currentframe()          # define a frame to track
#gpu_tracker = MemTracker(frame) 
for i in range(iterations):
    """
    compute lamda, loss1 and loss2
    
    """
  
    lamda = step_size(lamda0 = 0.2, opt = 1000, rate1 = 0.997, rate2 = 0.995, iteration = i)
    
    if i%10 == 0:
        print('iteration',i)
        print('lamda',lamda)    
    
    
    #gpu_tracker.track()
    loss1, g1 = model_gram(input_img.detach(), ref.detach())
    
    if i%10 == 0:
        print('loss1',loss1)
        #plt.hist(g1.cpu(),1000)
        #plt.show()
        print('g1',g1.max(),g1.min(),torch.mean(torch.abs(g1)))
    

   
   
    
    
    if i > 0:
        prev_loss = loss2
    
    #gpu_tracker.track()   
    loss2, g2 = mse(input_img.detach(), ref.detach())
    
    if i%10 == 0:
        print('\n')
        print('loss2',loss2)
        #plt.hist(g2.cpu(),1000)
        #plt.show()
        print('g2',g2.max(),g2.min(),torch.mean(torch.abs(g2)))
    

    
    """
    early stopping
    
    """
    
    if torch.abs(loss2-prev_loss) < 5e-5*loss2 or torch.abs(loss2-prev_loss)  < 5e-5:
        count += 1
    else:
        count = 0
    
    if count > 400:
        print('early stopping!!')
        break


    
    #gpu_tracker.track()
    y, comp = search_grad(ref.detach(), g = g2, gkeep = g1, img = input_img.detach(), mkeep = model_gram, tracker = None, lamda = lamda)
    
    cu = cu + comp
    
    if i %10 == 0:
        print('\n')
        print('cumulate comp:', cu)
        plt.figure()
        imshow(torch.clamp(y,0,1))
        print('\n\n\n')
    if comp > 1:
        print('too big step size, change lamda!!')
        torch.save(y,'temp.pt')  
        break
    
      
    input_img = y
    
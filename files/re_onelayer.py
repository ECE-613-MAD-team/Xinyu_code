#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:57:12 2019

@author: x227guo
"""

from models import *
from opt import *
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

import time
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc = 3
imsize = 256




iterations = 10000




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
    plt.savefig('test.jpg')
    
    plt.show()

    
def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('pebbles_noise6_3.jpg')
    plt.show()
# plt.figure()
# imshow(ref_img, title='reference texture')
ref_img = image_loader("./data/texture/pebbles.jpg")
ref_img = ref_img[:,:,0:256,0:256]
#imgn = image_loader("./data/texture/jpeg_10_radish.jpg")


"""

gaussian blur


"""

# imgn = cv2.imread("./data/texture/pebbles.jpg")
# imgn = cv2.GaussianBlur(imgn, ksize = (0, 0),sigmaX = (6.5))


# imgn = cv_converter(imgn)

"""

gaussian noise


"""

k = 8
ref = ref_img * 255
#noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
imgn = (noise) / 255
imgn = torch.clamp(imgn,0,1)



# plt.figure()
# imshow1(imgn, title='distorted image')
# plt.show()



#print(model_style)

imgn.data.clamp_(0,1)
input_img = imgn.detach()
#input_img = torch.load('temp.pt')

ref = ref_img.detach()

iters = 10
prev_loss2 = 0
count = 0
#lamda2 = -0.01
lamda = 0.02
#frame = inspect.currentframe()          # define a frame to track
#gpu_tracker = MemTracker(frame)
start = time.time()

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

theta_0 = 0
m_t = 0 
v_t = 0 

for i in range(iterations):
    loss2, g2 = one_layer(input_img.detach(), ref.detach())
    
    if i == 0:
       fix = lamda*torch.norm(g2) 
       #fix = torch.load('temp_fix.pt')*0.2
   
    #lamda = fix/torch.norm(g2) 
    #lamda = step_size(lamda0 = lamda, opt = 500, rate1 = 0.999, rate2 = 0.999, iteration = i)
    
    m_t = g2     # consider 90% of previous, and 10% of current
    v_t = (g2*g2) # 99.9% of previous (square grad), and 1% of current
    #m_cap = m_t/(1-(beta_1**(i+1)))      #calculates the bias-corrected estimates
    #v_cap = v_t/(1-(beta_2**(i+1)))      #calculates the bias-corrected estimates
    
    xm = input_img.flatten() - (lamda*m_t)/(torch.sqrt(v_t)+epsilon)
    #xm = torch.sub(input_img.flatten(), torch.mul(lamda, g2))
     
    xm = torch.clamp(xm, 0, 1)
    dim = torch.clamp((xm-ref.flatten()), -1, 1)
    xm = ref.flatten() + dim
    
    xm = xm.reshape(1,nc,imsize,imsize)
    if i%iters == 0:
        print('iteration',i)
        print('lamda',lamda)
        print('loss2',loss2)
        print('g2',g2.max(),g2.min(),torch.mean(torch.abs(g2)))
        imshow(torch.clamp(xm,0,1))


    input_img = xm



































#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 20:57:12 2019

@author: x227guo
"""

from PIL import Image
from models import *
import time
from one_layer import *

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)

ref_img = image_loader("./data/texture/pebbles.jpg")
ref_img = ref_img[:,:,0:256,0:256]

k = 10
ref = ref_img * 255
#noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
imgn = (ref+noise) / 255
imgn = torch.clamp(imgn,0,1)
imgn.data.clamp_(0,1)

m0 = torch.tensor([0]).float()


iters = 100


#################################################################### 4
start = time.time()

for i in range(iters):
   model_gram_forward(imgn,ref)

end = time.time()

print('\nmodel_gram_forward time',end-start,'s\n')

#################################################################### 1
start = time.time()

for i in range(iters):
   one_layer_forward(imgn,ref)

end = time.time()

print('\none_layer_forward time',end-start,'s\n')





#################################################################### 2
start = time.time()

for i in range(iters):
   one_layer(imgn,ref)

end = time.time()

print('\none_layer time',end-start,'s\n')


#################################################################### 3
start = time.time()

for i in range(iters):
   one_layer_opt(m0,imgn,ref)

end = time.time()

print('\none_layer_opt time',end-start,'s\n')





#################################################################### 5 
start = time.time()

for i in range(iters):
   model_gram(imgn,ref)

end = time.time()

print('\nmodel_gram time',end-start,'s\n')


#################################################################### 6
start = time.time()

for i in range(iters):
   model_gram_opt(m0,imgn,ref)

end = time.time()

print('\nmodel_gram_opt time',end-start,'s\n')






























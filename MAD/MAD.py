import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import pytorch_ssim
import numpy as np
import copy
import cv2


from models import *
from opt import *
"""

brick_wall.jpg
lacelike.jpg
pebbles.jpg
radish.jpg
red-peppers.jpg

einstein.png

"""




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
nc = 1


cu = 0
iterations = 1000
lamda = 0.2



loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
cnn = models.vgg19(pretrained=True).features.to(device).eval()





def cv_converter(img):
    image = Image.fromarray(img[...,::-1])
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)    
 
    
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)




def imshow(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.savefig('pebbles_noise_2.jpg',dpi = 300)
    plt.show()

    
def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    #plt.savefig('pebbles_noise_3.jpg',dpi = 300)
    plt.show()
# plt.figure()
# imshow(ref_img, title='reference texture')

ref_img = image_loader("./data/texture/einstein.png")

"""

gaussian blur


"""

# imgn = cv2.imread("./data/texture/pebbles.jpg")
# imgn = cv2.GaussianBlur(imgn, ksize = (0, 0),sigmaX = (6.5))


# imgn = cv_converter(imgn)

"""

gaussian noise


"""
m = 128
ref_img = ref_img[:,:,32:32+m,32:32+m]
#print(ref_img.shape)
k = 8
ref = ref_img * 255
#noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
noise = torch.randn(1,nc,m,m)*torch.sqrt((torch.tensor([2.0])**k)) 
imgn = (ref+noise.to(device)) / 255
imgn = imgn[:,:,32:32+m,32:32+m]
imgn = torch.clamp(imgn,0,1)



# plt.figure()
# imshow1(imgn, title='distorted image')
# plt.show()



model_style, style_losses = get_style_model_and_losses(cnn,
          cnn_normalization_mean, cnn_normalization_std, ref_img)

print(model_style)

imgn.data.clamp_(0,1)
input_img = imgn.detach()
ref = ref_img.detach()

for i in range(iterations):
    if i%10 == 0:
        lamda = lamda*0.8
    #loss1, g1 = model_gram(model_style, input_img.detach(), style_losses)
    #loss1, g1 = mse(input_img.detach(), ref.detach())
    loss1, g1 = mse(input_img.detach(), ref.detach())
    
    if i%10 == 0:
        print('loss1',loss1)
        #plt.hist(g1.cpu(),1000)
        #plt.show()
        print('g1',g1.max(),g1.min(),torch.mean(torch.abs(g1)))
    

    #loss2, g2 = mse(input_img.detach(), ref.detach())
    #loss2, g2 = model_gram(model_style, input_img.detach(), style_losses)
    loss2, g2 = ssim(input_img.detach(), ref.detach())
    
    
    if i%10 == 0:
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
    
    
    if i%10 == 0:
        print('lamda:', lamda)
    g_prev = g2
    y, comp = search_grad(ref.detach(), g = g2, gkeep = g1, img = input_img.detach(), mkeep = mse, mkeep_opt = mse_opt, lamda = lamda)

    cu = cu + comp
    
    if i %10 == 0:
        print('cumulate comp:', cu)
        plt.figure()
        imshow(torch.clamp(y,0,1))
    if comp > 1:
        torch.save(y,'temp.pt')  
        break
    
    
        
    input_img = y
    

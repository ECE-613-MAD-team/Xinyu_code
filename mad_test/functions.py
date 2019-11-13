import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_ssim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

from models import *

# TODO: to use names to control testing objects

einstein = 'einstein'
pebbles = 'pebble'
image_tag = einstein

"""

brick_wall.jpg
lacelike.jpg
pebbles.jpg
radish.jpg
red-peppers.jpg

"""

# functions

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

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.clone().detach().requires_grad_(True).view(-1, 1, 1)
        self.std = std.clone().detach().requires_grad_(True).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# show and save
def imshow(tensor, title=None):
    tensor = torch.clamp(tensor,0,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.savefig(image_tag+'_result.jpg',dpi = 300)
    plt.show()

def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image, cmap='gray')
    if title is not None:
        plt.title(title)
    plt.savefig(image_tag+'_origin.jpg', dpi = 300)
    plt.show()

# noise

"""

Gaussian
Blur
sgan4
jpeg

"""

def gaussian_noise(k=8):
    noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
    return noise

# mad search

def bisection(mkeep, lower, upper, g, ref, y_n, xm):
    y_n_loss, _ = mkeep(y_n, ref)
    a = lower
    b = upper
    m = (a+b)/2
    while b - a > 1e-8:
        loss_a = mkeep(xm+a*g, ref)[0] - y_n_loss
        loss_m = mkeep(xm+m*g, ref)[0] - y_n_loss
        loss_b = mkeep(xm+b*g, ref)[0] - y_n_loss
        if loss_a * loss_b > 0 and abs(loss_b) - abs(loss_a) > 0:
            b = -b
            m = (a+b)/2
        elif loss_a * loss_b > 0 and abs(loss_b) - abs(loss_a) <= 0:
            b = 2 * (b - a)
            m = (a+b)/2
        elif loss_m * loss_a <= 0:
            b = m
            m = (a+b)/2
        elif loss_m * loss_b <= 0:
            a = m
            m = (a+b)/2
        else:
            print('a wider bound!')
            return None, None
    m = (a+b)/2
    comp = mkeep(xm+m*g, ref)[0] - y_n_loss
    return comp, (xm + m*g)

def search_grad(ref, g_1n, g_2n, img = None, mkeep = None, lamda = None, iterate = None):
    # mad hold loss
    r = 0.5 - iterate * 0.001
    step = 5e-5 # for control of step
    N = r / step
    vsearch = np.linspace(r, 0, N+1)

    y_n = img # current image

    g_n = g_2n - torch.mul(torch.div(torch.dot(g_2n, g_1n), torch.dot(g_1n, g_1n)), g_1n)

    #y_n_prime = torch.sub(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    y_n_prime = torch.add(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    # sub or add depends on the maximal or minimal opt goal
    #print('y_n_prime - y_n: ', (y_n_prime - y_n).sum() )
    
    y_n_loss, _ = mkeep(y_n.detach(), ref.detach())
    # mkeep is used to calculate the loss from the holding method
    # loss from two gradient
    y_n_prime_loss, g_1n_prime = mkeep(y_n_prime.detach(), ref.detach())
    
    comp = torch.abs(y_n_prime_loss - y_n_loss)
    first_comp = comp
    #print('comp', comp) # current loss error for the holding method

    g_1n_prime_bi = mkeep(y_n_prime.detach(), ref.detach())[1].reshape(1,nc,imsize,imsize)
    comp, y_n1 = bisection(mkeep, -1, 0, g_1n_prime_bi, ref, y_n, y_n_prime)

    # if comp == None:
    #     y_n1 = y_n_prime
    #     for i,v in enumerate(vsearch):
    #         tep_img = y_n1.flatten() + v * g_1n_prime
    #         tep_img = tep_img.reshape(1, nc, imsize, imsize)
    #         tep_mkeep_loss, _ = mkeep(tep_img.detach(), ref.detach())
    #         tep_comp =  torch.abs(tep_mkeep_loss - y_n_loss)
            
    #         # if i % 1000 == 0:
    #         #     print('Current v: ' + str(v) + ', temp_comp: ' + str(tep_comp))
    #         if tep_comp  < comp:
    #             # v is correct
    #             comp = tep_comp
    #             y_n1 = tep_img
                
    #             if tep_comp < 5e-5:
    #                 print("find one!")
    #                 break
    #         # else do not renew yn, just reduce v        

    #         # For -v:
    #         tep_img = y_n1.flatten() - v * g_1n_prime
    #         tep_img = tep_img.reshape(1, nc, imsize, imsize)
    #         tep_mkeep_loss, _ = mkeep(tep_img.detach(), ref.detach())
    #         tep_comp =  torch.abs(tep_mkeep_loss - y_n_loss)
            
    #         # if i % 1000 == 0:
    #         #     print('Current v: ' + str(v) + ', temp_comp: ' + str(tep_comp))
    #         if tep_comp  < comp:
    #             # v is correct
    #             comp = tep_comp
    #             y_n1 = tep_img
                
    #             if tep_comp < 5e-5:
    #                 print("find one!")
    #                 break

    return y_n1, first_comp
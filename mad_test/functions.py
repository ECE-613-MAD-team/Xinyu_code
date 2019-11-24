import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_ssim
import numpy as numpy
import kornia
import os
import torchvision

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models

from models import *
from io import StringIO
from io import BytesIO

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
    plt.savefig('distort_origin.jpg', dpi = 300)
    plt.show()

# noise

"""

Gaussian
Blur
sgan4
jpeg

"""

def gaussian_noise(img, k=8):
    noise = torch.randn(1, nc, imsize, imsize) * torch.sqrt( (torch.tensor([2.0])**k) ) 
    imgn = ( img + noise.to(device) ) / 255
    imgn = torch.clamp(imgn, 0, 1)
    return imgn

def blur_noise(img, kernal=3, sigma=6):
    gauss = kornia.filters.GaussianBlur2d( (kernal, kernal), (sigma, sigma) )
    imgn = gauss(img) / 255
    imgn = torch.clamp(imgn, 0, 1)

    return imgn

def jpeg_noise(img):
    img /= 255
    img = torch.clamp(img, 0, 1)
    temp = img.cpu().clone() 
    temp = temp.squeeze(0) 
    temp = unloader(temp)
    noise_os = os.path.join('./jpeg_noise.jpeg')
    temp.save(noise_os, "JPEG", quality=25)
    imgn = image_loader(noise_os)
    # finally give up using buffer ...
    #buffer = BytesIO()
    # buffer.seek(0)
    # contents = buffer.getvalue()
    # image = Image.frombuffer("L", (imsize, imsize), contents, 'raw', "L", 0, 1)
    # image = loader(image).unsqueeze(0)
    return imgn

def gamma_noise(img, gamma=2):
    img /= 255
    img = torch.clamp(img, 0, 1)
    temp = img.cpu().clone() 
    temp = temp.squeeze(0) 
    temp = unloader(temp)
    image = torchvision.transforms.functional.adjust_gamma(temp, gamma, gain=1)
    image = loader(image).unsqueeze(0)
    image = torch.clamp(image, 0, 1)
    return image.to(device, torch.float)

# mad search

def prof_wang(mkeep, ref, xm, lamda2, gkeep, init_loss):
    # citation:
    mb, _ = mkeep(xm ,ref)
    
    temp_im = xm + lamda2*gkeep
    mt, _ = mkeep(temp_im.detach() ,ref)
    lamda2 = (lamda2*(init_loss - mb)/(mt - mb + 1e-6)).detach()
    xk = xm + lamda2*gkeep
    mk, _ = mkeep(xk.detach() ,ref)
    comp = mk-init_loss
    y = xk.reshape(1,nc,imsize,imsize)
    return comp, y, lamda2

def Adam(m0, xm, ref, mkeep_opt):    
    xm = xm.reshape(1,nc,imsize,imsize)
    lr = 2e-5  # vgg+gram 2e-5
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    m_t = 0 
    v_t = 0 
    t = 0
    
    comp = 10
    while comp > 1e-4: #vgg+gram 1e-6mm0
        t += 1
        if t > 1e3:
            break
        #print('t',t)
        #if t > 10: 
        #    lr = lr*0.9
        comp, g_t = mkeep_opt(m0,xm,ref)
        m_t = beta_1*m_t + (1-beta_1)*g_t     # consider 90% of previous, and 10% of current
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) # 99.9% of previous (square grad), and 1% of current
        m_cap = m_t/(1-(beta_1**t))      #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))      #calculates the bias-corrected estimates
        
        #xm_prev = xm
        xm = xm - (lr*m_cap)/(torch.sqrt(v_cap)+epsilon)

    return comp, xm

def bisection(mkeep, lower, upper, g, ref, y_n, xm):
    y_n_loss, _ = mkeep(y_n, ref)
    a = lower
    b = upper
    m = (a+b)/2
    t = 0
    while b - a > 1e-5:
        t += 1
        if t > 1e2:
            break
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

def bisection1(f, lower, upper, g, ref, init_loss, xm):
    xm = xm.reshape(1,nc,imsize,imsize)
    obj = init_loss
    var = 1
    a = lower
    b = upper
    m = (a+b)/2
    flag = 0
    m1, _ = f((xm+a*g),ref)
    m2, _ = f((xm+m*g),ref)
    m3, _ = f((xm+b*g),ref)
    tol = 30
    x = 0.01
    
    while var == 1:            
        if (m3-m2) <= 0  or (m2-m1) <= 0:
            a = m
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref)
            m2, _ = f((xm+m*g),ref)
            #m3, _ = f(xm+b*g,ref)
            if flag > tol :
                #print('!!!!!!!!!!!')
                break
            else:
                flag += 1
                continue
        
        if (m1-obj) > 0 and (m3-obj) > 0: 
            a = a-x
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref)
            m2, _ = f((xm+m*g),ref)
            #m3, _ = f(xm+b*g,ref)
            if flag > tol :
                #print('!!!!!!!!!!!')
                break
            else:
                flag += 1
                continue
        elif (m1-obj) < 0 and (m3-obj) < 0:
            b = b+x
            m = (a+b)/2
            #m1, _ = f(xm+a*g,ref)
            m2, _ = f((xm+m*g),ref)
            m3, _ = f((xm+b*g),ref)
            if flag > tol :
               # print('!!!!!!!!!!!')
                break
            else:
                flag += 1
                continue
        else:
            pass
        
        if (m3-obj) < 0 or (m1-obj) > 0:
            continue
        
        if (m1-obj)*(m2-obj) <= 0:
            b = m
            m = (a+b)/2
            m2, _ = f((xm+m*g),ref)
            m3, _ = f((xm+b*g),ref)
        elif (m2-obj)*(m3-obj) <= 0:
            a = m
            m = (a+b)/2
            m1, _ = f((xm+a*g),ref)
            m2, _ = f((xm+m*g),ref)
        elif flag > tol :
            break
        else:
            pass
         
        if b-a < 1e-6:
             break
        
    comp = m2-obj
       
    return comp, (xm + m*g)

def search_grad(ref, g_1n, g_2n, direction, img = None, mkeep = None, mkeep_opt = None, lamda = None, init_loss = None, lamda2 = None):

    y_n = img # current image

    g_n = g_2n - torch.mul(torch.div(torch.dot(g_2n, g_1n), torch.dot(g_1n, g_1n)), g_1n)

    #y_n_prime = torch.sub(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    if direction == 0:
        y_n_prime = torch.add(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    elif direction == 1:
        y_n_prime = torch.sub(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)

    # citation
    ##############################################
    y_n_prime = torch.clamp(y_n_prime, 0, 1)
    dim = torch.clamp((y_n_prime-ref), -1, 1)
    y_n_prime = ref + dim
    ################################################

    # sub or add depends on the maximal or minimal opt goal
    #print('y_n_prime - y_n: ', (y_n_prime - y_n).sum() )
    
    y_n_loss, _ = mkeep(y_n.detach(), ref.detach())
    # mkeep is used to calculate the loss from the holding method
    # loss from two gradient
    y_n_prime_loss, _ = mkeep(y_n_prime.detach(), ref.detach())

    comp = torch.abs(y_n_prime_loss - y_n_loss)
    first_comp = comp
    #print('comp', comp) # current loss error for the holding method

    g_1n_prime_bi = mkeep(y_n_prime.detach(), ref.detach())[1].reshape(1,nc,imsize,imsize)
    #comp, y_n1 = bisection(mkeep, -5, 0, g_1n_prime_bi, ref, y_n, y_n_prime)
    comp, y_n1, lamda2_prime = prof_wang(mkeep, ref, y_n_prime.detach(), lamda2, g_1n_prime_bi, init_loss)
    if torch.abs(comp) > 0.01:
        comp, y_n1 = bisection(mkeep, -5, 0, g_1n_prime_bi, ref, y_n, y_n_prime)
        #comp, y_n1 = bisection1(mkeep, -0.1, 0.1, g_1n_prime_bi, ref, init_loss, y_n_prime)
    if torch.abs(comp) > 0.01:
        print('enter adam')
        comp, y_n1 = Adam(init_loss.detach(), y_n_prime, ref, mkeep_opt = mkeep_opt)

    #comp, y_n1 = Adam(init_loss.detach(), y_n_prime, ref, mkeep_opt = mkeep_opt)



    # gonna add adam to this

    return y_n1, first_comp, lamda2_prime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_ssim
from torch.autograd import Variable

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

from math import exp
import numpy as np
import copy
import cv2
import datetime

# TODO: using constant to control type
weight_mse = 2e4
weight_gram = 1e5
weight_ssim = 2e2

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
nc = 1

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
cnn = models.vgg19(pretrained=True).features.to(device).eval()

unloader = transforms.ToPILImage()  # reconvert into PIL image

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

# class ContentLoss(nn.Module):

#     def __init__(self, target,):
#         super(ContentLoss, self).__init__()
#         # we 'detach' the target content from the tree used
#         # to dynamically compute the gradient: this is a stated value,
#         # not a variable. Otherwise the forward method of the criterion
#         # will throw an error.
#         self.target = target.detach()

#     def forward(self, input):
#         self.loss = F.mse_loss(input, self.target)
#         return input

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

"""


texture network



"""

#content_layers_default = ['conv_4']
style_layers_default = ['conv_1','pool_2', 'pool_4', 'pool_8', 'pool_12']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, 
                               #content_img,
                               #content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    #content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

#         if name in content_layers:
#             # add content loss:
#             target = model(content_img).detach()
#             content_loss = ContentLoss(target)
#             model.add_module("content_loss_{}".format(i), content_loss)
#             content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break
         #print(i)
         if isinstance(model[i], StyleLoss):
             break

    model = model[:(i + 1)]

    return model, style_losses     #, content_losses

# def get_input_optimizer(input_img):
#     # this line to show that input is a parameter that requires a gradient
#     optimizer = optim.LBFGS([input_img.requires_grad_()])
#     return optimizer


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
# plt.figure()
# imshow(ref_img, title='reference texture')

def model_gram(model, img, losses):
    
    weight = weight_gram = 1e5 # control the gradient
    img.requires_grad_()
    model(img)
    
    style_score = 0
    for sl in style_losses:
        style_score += weight * sl.loss
    style_score.backward()
    
    return style_score, img.grad.flatten()

def mse(img, ref):
    # MSR loss
    weight = weight_mse
    
    img.requires_grad_()
    
    N = nc * imsize * imsize
    loss = weight * ( (img - ref)**2 ).sum() / N
    loss.backward()

    return loss, img.grad.flatten()

def ssim(img, ref):
    img.requires_grad_()
    
    ssim_value = pytorch_ssim.ssim(ref, img)
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_out = -weight_ssim * ssim_loss(ref, img)
    ssim_out.backward()
    
    return ssim_value, img.grad.flatten()

def search_grad(ref, g_1n, g_2n, img = None, mkeep = None, lamda = None, iterate = None):
    # mad hold loss
    r = 0.5 - iterate * 0.001
    step = 5e-5 # for control of step
    N = r / step
    vsearch = np.linspace(r, 0, N+1)
    y_n = img # current image

    g_n = g_2n - torch.mul(torch.div(torch.dot(g_2n, g_1n), torch.dot(g_1n, g_1n)), g_1n)
    # plt.hist(g_n.cpu(), 1000) #???
    # plt.show()
    # print('g_n: ', g_n.max(), g_n.min(), torch.mean(torch.abs(g_n)))

    y_n_prime = torch.sub(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    #y_n_prime = torch.add(y_n.flatten(), torch.mul(lamda, g_n)).reshape(1, nc, imsize, imsize)
    # sub or add depends on the maximal or minimal opt goal
    print('y_n_prime - y_n: ', (y_n_prime - y_n).sum() )
    
    y_n_loss, _ = mkeep(y_n.detach(), ref.detach())
    # mkeep is used to calculate the loss from the holding method
    # loss from two gradient
    y_n_prime_loss, g_1n_prime = mkeep(y_n_prime.detach(), ref.detach())
    
    #print('g_1n_prime', g_1n_prime.max(), g_1n_prime.min(), torch.mean(torch.abs(g_1n_prime)))
    comp = torch.abs(y_n_prime_loss - y_n_loss)
    first_comp = comp
    print('comp', comp) # current loss error for the holding method
    
    y_n1 = y_n_prime
    for i,v in enumerate(vsearch):
        tep_img = y_n1.flatten() + v * g_1n_prime
        tep_img = tep_img.reshape(1, nc, imsize, imsize)
        tep_mkeep_loss, _ = mkeep(tep_img.detach(), ref.detach())
        tep_comp =  torch.abs(tep_mkeep_loss - y_n_loss)
        
        # if i % 1000 == 0:
        #     print('Current v: ' + str(v) + ', temp_comp: ' + str(tep_comp))
        if tep_comp  < comp:
            # v is correct
            comp = tep_comp
            y_n1 = tep_img
            
            if tep_comp < 5e-5:
                print("find one!")
                break
        # else do not renew yn, just reduce v        

        # For -v:
        tep_img = y_n1.flatten() - v * g_1n_prime
        tep_img = tep_img.reshape(1, nc, imsize, imsize)
        tep_mkeep_loss, _ = mkeep(tep_img.detach(), ref.detach())
        tep_comp =  torch.abs(tep_mkeep_loss - y_n_loss)
        
        # if i % 1000 == 0:
        #     print('Current v: ' + str(v) + ', temp_comp: ' + str(tep_comp))
        if tep_comp  < comp:
            # v is correct
            comp = tep_comp
            y_n1 = tep_img
            
            if tep_comp < 5e-5:
                print("find one!")
                break

    return y_n1, first_comp

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    # load images
    ref_img = image_loader("./pic_src/"+image_tag+".png")

    # k = 8
    # ref = ref_img.detach()
    # noise = torch.randn(1, nc, imsize, imsize) * torch.sqrt( (torch.tensor([2.0])**k) ) / torch.tensor(255.0) 
    # imgn = (ref+noise.to(device)) / 255
    # imgn = torch.clamp(imgn, 0, 1)

    # m = 128
    # ref_img = ref_img[:,:,32:32+m,32:32+m]
    #print(ref_img.shape)
    k = 8
    ref = ref_img * 255
    noise = torch.randn(1,nc,imsize,imsize)*torch.sqrt((torch.tensor([2.0])**k)) 
    #noise = torch.randn(1,nc,m,m)*torch.sqrt((torch.tensor([2.0])**k)) 
    print(ref.shape)
    print(noise.to(device).shape)
    imgn = (ref+noise.to(device)) / 255
    imgn = torch.clamp(imgn,0,1)

    print(ref.detach().shape)

    """

    Gaussian
    Blur
    sgan4
    jpeg

    """
    plt.figure()
    imshow1(imgn, title='distorted image')

    model_style, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, ref_img)
    print('model load success!')
    # print(model_style)

    imgn.data.clamp_(0,1) # transition from [0,255] -> (0,1)?
    input_img = imgn.detach()
    ref = ref_img.detach()
    cu = 0

    # mad search find the maximal / minimal
    iterations = 100
    for i in range(iterations):
        lamuda = 1 - i * 0.005
        #loss1, g1 = model_gram(model_style, input_img.detach(), style_losses)
        loss1, g1 = mse(input_img.detach(), ref.detach())
        
        #loss2, g2 = mse(input_img.detach(), ref.detach())
        #loss2, g2 = model_gram(model_style, input_img.detach(), style_losses)
        loss2, g2 = ssim(input_img.detach(), ref.detach())
        
        y, comp = search_grad(ref.detach(), g_2n = g2, g_1n = g1, img = input_img.detach(), mkeep = mse, lamda = lamuda, iterate = i)
        
        cu = cu + comp
        loss_keep, _ = mse(y.detach(), ref.detach())
        loss_change, _ = ssim(y.detach(), ref.detach())
        print('iteration: ' + str(i))
        print('loss1:     ' + str(loss1))
        print('loss2:     ' + str(loss2))
        print('keep:      ' + str( torch.abs(loss_keep - loss1) ) )
        print('change:    ' + str( torch.abs(loss_change - loss2) ) )
        #print('cumulate comp:', cu)
        # if comp > 1:
        #     torch.save(y,'temp.pt')  
        #     break
        
        # if i % 10 == 0:
        #     print('iterate: ' + str(i))
        #     plt.figure()
        #     imshow(torch.clamp(y,0,1))
            
        input_img = y
        if comp < 5e-5:
            print('iterate: ' + str(i))
            print('comp: ' + str(comp))
            # plt.figure()
            # imshow(torch.clamp(y, 0, 1))
            break

    # torch.save(input_img, 'pebbles_noise_1.pt')
    # plt.figure()
    plt.figure()
    imshow(torch.clamp(input_img, 0, 1))
    endtime = datetime.datetime.now()
    print ((endtime - starttime).seconds)

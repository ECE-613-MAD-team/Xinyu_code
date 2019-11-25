import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_ssim
import time

import torchvision.models as models

#content_layers_default = ['conv_4']
#style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
style_layers_default = ['conv_1','pool_2', 'pool_4', 'pool_8', 'pool_12']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG19 model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
cnn = models.vgg19(pretrained=True).features.to(device).eval()

#constants
weight_mse = 2e4
weight_gram = 5e4
weight_ssim = 2e3
imsize = 256
nc = 3

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

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

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

def model_gram(img, ref=None, model_style=None, style_losses=None):
    
    img = img.reshape(1, nc, imsize, imsize)
    
    img.requires_grad_()
    # model_style, style_losses = get_style_model_and_losses(cnn,
    #                             cnn_normalization_mean, cnn_normalization_std, ref)
    
    model_style(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_gram * sl.loss
    style_score.backward()
    grad = img.grad
    
    return style_score, grad.flatten()

def model_gram_opt(m0, img, ref=None, model_style=None, style_losses=None):

    img = img.reshape(1, nc, imsize, imsize)
    
    img.requires_grad_()
    # model_style, style_losses = get_style_model_and_losses(cnn,
    #                             cnn_normalization_mean, cnn_normalization_std, ref)
    
    model_style(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_gram * sl.loss
    comp = (m0-style_score)**2
    
    comp.backward()
    
    grad = img.grad
    
    return comp, grad

def mse(img, ref, model_style=None, style_losses=None):
    _, nc, imsize, imsize = img.shape
    img.requires_grad_()
    
    N = nc*imsize*imsize
    loss = weight_mse*((img-ref)**2).sum() / (N)
    loss.backward()

    return loss, img.grad.flatten()

def mse_opt(m0, temp, ref, model_style=None, style_losses=None):
    temp = temp.reshape(1, nc, imsize, imsize)
    temp.requires_grad_()

    N = nc * imsize * imsize
    loss_mse = weight_mse * ((temp - ref) ** 2).sum() / (N)
    comp = (m0 - loss_mse) ** 2
    comp.backward()

    return comp, temp.grad

def ssim(img, ref, model_style=None, style_losses=None):

    img.requires_grad_()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_value = ssim_loss(ref, img)
    ssim_out = -weight_ssim * ssim_value
    ssim_out.backward()
    grad_return = img.grad.flatten()

    return ssim_value, grad_return

def ssim_opt(m0, temp, ref, model_style=None, style_losses=None):
    temp = temp.reshape(1, nc, imsize, imsize)

    # _, nc, imsize, imsize = temp.shape
    temp.requires_grad_()

    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    ssim_out = -weight_ssim * ssim_loss(ref, temp)
    comp = ((-weight_ssim * m0) - ssim_out) ** 2
    comp.backward()

    return comp, temp.grad
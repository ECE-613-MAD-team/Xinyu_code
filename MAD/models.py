import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_ssim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import time


#content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
#style_layers_default = ['conv_1','pool_2', 'pool_4', 'pool_8', 'pool_12']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_feature_maps = 128
scales = [3, 5, 7, 11, 15, 23, 37, 55]
imsize = 256
nc = 3

weight_mse = 2e4
weight_gram = 2e3
weight_ssim = 2e3
weight_onelayer = 5e11


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
cnn = models.vgg19(pretrained=True).features.to(device)

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image
def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)
#ref_img = image_loader("./data/texture/pebbles.jpg")
 
class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        
        G = gram_matrix(input)
        start = time.time()
        a,b = G.shape
        N = a*b
        #self.loss = F.mse_loss(G, self.target) 
        #print('a',self.loss)
        self.loss = ((G-self.target)**2).sum() / N
        #print('b',self.loss)
        end = time.time()
        #print('time_1_2 forward passing',end-start,'s')
        return input



def gram_matrix(input):
  
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.

    return G.div(a * b * c * d)

"""


texture network



"""


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
        
class Onelayer_Net(nn.Module):
    
    def __init__(self,scales,n_feature_maps):
        super(Onelayer_Net,self).__init__()
        
        self.multiple_scales = [nn.Conv2d(nc, n_feature_maps, filter_size, 1, filter_size//2).to(device)
                       for filter_size in scales]

        self.nonliners = [nn.ReLU().to(device) for filter_size in scales]

    def forward(self,x):
        
        out =  [conv(x) for conv in self.multiple_scales]
        out =  [f(out[i]) for i,f in enumerate(self.nonliners)]
        out =  torch.cat(out,1)
        
        return out


def get_onelayer_model_and_losses(normalization_mean, normalization_std,
                                  style_img,
                                  device):
                           
    net = Onelayer_Net(scales,n_feature_maps)
    

    style_img = style_img.to(device)
    
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    model = nn.Sequential(normalization)
    model.add_module('onelayer',net)
    style_losses = []
    

    target_feature = model(style_img).detach()

    style_loss = StyleLoss(target_feature)
    model.add_module('style_loss',style_loss)
    
    style_losses.append(style_loss)
    
    
   
    #del style_img
    #start = time.time()
    #torch.cuda.empty_cache()
    #end = time.time()
    #print('1_1',end-start,'s')

    return model,style_losses

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img,
                                device,
                               #content_img,
                               #content_layers=content_layers_default,
                               style_layers=style_layers_default):
    
    style_img = style_img.to(device)
    
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
    
    #print(model)
    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
#         if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
#             break
         #print(i)
         if isinstance(model[i], StyleLoss):
             break

    model = model[:(i + 1)]
    
   
    del style_img
    torch.cuda.empty_cache()

    return model, style_losses     #, content_losses

#model_style, style_losses = get_style_model_and_losses(cnn,
#          cnn_normalization_mean, cnn_normalization_std, ref_img, device = device)

def model_gram_forward(img, ref):
    
    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    
    
    
    model_style, style_losses = get_style_model_and_losses(cnn,
          cnn_normalization_mean, cnn_normalization_std, ref, device)
    
    with torch.no_grad():
        model_style(img)
        
    style_score = 0
    for sl in style_losses:
        style_score += weight_gram*sl.loss
   
    grad = 0
    
    del ref,img,model_style
    torch.cuda.empty_cache()
    
    return style_score.cpu(), grad


def model_gram(img, ref):
    
    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    
    img.requires_grad_()
    
    model_style, style_losses = get_style_model_and_losses(cnn,
          cnn_normalization_mean, cnn_normalization_std, ref, device)
    
    model_style(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_gram*sl.loss
    style_score.backward()
    grad = img.grad.cpu()
    
    del ref,img,model_style
    torch.cuda.empty_cache()
    return style_score.cpu(), grad.flatten()

def one_layer_forward(img,ref):

    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    
    
   
    onelayer, style_losses = get_onelayer_model_and_losses(
          cnn_normalization_mean, cnn_normalization_std, ref, device)

    
    with torch.no_grad():
        onelayer(img)


    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer*sl.loss
    
    grad = 0
    
    del ref,img,onelayer
    torch.cuda.empty_cache()


    return style_score.cpu(), grad



def one_layer(img,ref):

    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    
    img.requires_grad_()
    
    onelayer, style_losses = get_onelayer_model_and_losses(
          cnn_normalization_mean, cnn_normalization_std, ref, device)

    onelayer(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer*sl.loss
    style_score.backward()
    grad = img.grad.cpu()
    
    del ref,img,onelayer
    torch.cuda.empty_cache()
    return style_score.cpu(), grad.flatten()


def one_layer_opt(m0,img,ref):

    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    m0 = m0.to(device)
    
    img.requires_grad_()
    

    onelayer, style_losses = get_onelayer_model_and_losses(
          cnn_normalization_mean, cnn_normalization_std, ref, device)

    onelayer(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_onelayer*sl.loss
        
    comp = (m0-style_score)**2
    
    comp.backward()
    grad = img.grad.cpu()
    
    del ref,img,onelayer
    torch.cuda.empty_cache()
    return comp.cpu(), grad


def model_gram_opt(m0, img, ref):

    img = img.reshape(1,nc,imsize,imsize)
    img = img.to(device)
    ref = ref.to(device)
    m0 = m0.to(device)
    
    img.requires_grad_()
    
    model_style, style_losses = get_style_model_and_losses(cnn,
          cnn_normalization_mean, cnn_normalization_std, ref, device = device)
    
    model_style(img)
    style_score = 0
    for sl in style_losses:
        style_score += weight_gram*sl.loss
    comp = (m0-style_score)**2
    
    comp.backward()
    
    grad = img.grad.cpu()
    
    
    del ref,img,model_style
    torch.cuda.empty_cache()
    
    return comp.cpu(), grad


def mse(img,ref):

    
    img = img.reshape(1,nc,imsize,imsize)
    img.requires_grad_()
    
    N = nc*imsize*imsize
    loss = weight_mse*((img-ref)**2).sum() / (N)
    loss.backward()
#    
#    del ref
#    torch.cuda.empty_cache()
    return loss, img.grad.flatten()

def ssim(img, ref):
    
   
    img = img.reshape(1,nc,imsize,imsize)
    img.requires_grad_()
    
    ssim_value = pytorch_ssim.ssim(ref, img)
    ssim_loss = pytorch_ssim.SSIM()
    ssim_out = -weight_ssim*ssim_loss(ref, img)
    ssim_out.backward()
    
#    del ref
#    torch.cuda.empty_cache()
    return ssim_value, img.grad.flatten()


def mse_opt(m0, temp, ref):
    
    temp = temp.reshape(1,nc,imsize,imsize)
    temp.requires_grad_()
   
    N = nc*imsize*imsize
    loss_mse = weight_mse*((temp-ref)**2).sum() / (N)
    comp = (m0-loss_mse)**2
   # print('comp',comp,m0-loss_mse,m0,loss_mse)
    comp.backward()

    return comp, temp.grad
#
#
#
def ssim_opt(m0, temp, ref):
    
    temp = temp.reshape(1,nc,imsize,imsize)
   # _, nc, imsize, imsize = temp.shape
    temp.requires_grad_()
    
    ssim_value = pytorch_ssim.ssim(ref, temp)
    ssim_loss = pytorch_ssim.SSIM()
    ssim_out = -weight_ssim*ssim_loss(ref, temp)
    comp = ((-weight_ssim*m0)-ssim_out)**2
    comp.backward()
    
    return comp, temp.grad

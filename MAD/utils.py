import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

nc = 3
imsize = 256
loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor
unloader = transforms.ToPILImage()  # reconvert into PIL image


def step_size(lamda0 ,opt ,rate1 ,rate2 ,iteration):


    if iteration < opt:
        lamda = lamda0 *(rate1**iteration)
    else:
        lamda = lamda0 *(rate1**iteration) *(rate2**(iteration-opt))

    return lamda





def cv_converter(img):
    image = Image.fromarray(img[... ,::-1])
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(torch.float)




def imshow(tensor, title=None):
    tensor = torch.clamp(tensor ,0 ,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('dotted_0126_noise10_4.jpg')

    plt.show()


def imshow1(tensor, title=None):
    tensor = torch.clamp(tensor ,0 ,1)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig('pebbles_noise6_3.jpg')
    plt.show()
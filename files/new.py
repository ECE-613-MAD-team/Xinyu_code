import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from matplotlib.pyplot import imsave
import torchvision.transforms.functional as TF

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        
        G = gram_matrix(input)

   
        self.loss = F.mse_loss(G, self.target)

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


def shifted_gramm_matrix(x, shiftX=1, shiftY=0):
    X,Y = x.shape[2:4]
    features_pdelta = x[0,:,shiftX:,shiftY:].flatten()
    features_mdelta = x[0,:,:X-shiftX,:Y-shiftY].flatten()
    g = torch.dot(features_pdelta, features_mdelta)
    return g

# width range
def LR_flipped_gram_matrix(x):
    features = x[0,:,:,:].flatten()
    flipped_features = x[0,:,:,::-1].flatten()
    g = torch.dot(features, flipped_features)
    return g

# height range
def UD_flipped_gram_matrix(x):
    features = x[0,:,:,:].flatten()
    flipped_features = x[0,:,::-1,:].flatten()
    g = torch.dot(features, flipped_features)
    return g


def cross_correlation_loss(A, X, shiftX, shiftY):
    a = A[layer]
    x = X[layer]
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    A_x = shifted_gramm_matrix(a, shiftX, 0)
    G_x = shifted_gramm_matrix(x, shiftX, 0)
    A_y = shifted_gramm_matrix(a, 0, shiftY)
    G_y = shifted_gramm_matrix(x, 0, shiftY)
    
    loss = 1./(8 * N**2 * M**2)*(((G_x - A_x)**2) + (G_y - A_y)**2).sum()
    return loss


def flip_loss(A, X, layer):
    a = A[layer]
    x = X[layer]
    
    N = a.shape[1]
    M = a.shape[2] * a.shape[3]
    
    A_lr = LR_flipped_gram_matrix(a)
    G_lr = LR_flipped_gram_matrix(x)
    A_ud = UD_flipped_gram_matrix(a)
    G_ud = UD_flipped_gram_matrix(x)
    
    loss = 1./(8 * N**2 * M**2)*(((G_lr - A_lr)**2) + (G_ud - A_ud)**2).sum()
    return loss










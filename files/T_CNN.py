import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import copy

from models import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc = 3
imsize = 256



alexnet = models.alexnet(pretrained=True).to(device)
tcnn = alexnet.features[0:5]
ref_img = image_loader("./data/texture/" + '1' + ".jpg")
with torch.no_grad():
    _, _, pool_size, _= tcnn(ref_img).shape
global_avg_pool = nn.AvgPool2d(kernel_size = pool_size)
tcnn.add_module('global_avg_pool',global_avg_pool)


class TcnnLoss(nn.Module):
    def __init__(self, target_feature):
        super(TcnnLoss, self).__init__()
        self.target = target_feature.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def get_tcnn_model_and_losses(normalization_mean, normalization_std,
                              style_img,
                              device):
    net = copy.deepcopy(tcnn)

    style_img = style_img.to(device)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    model = nn.Sequential(normalization)
    model.add_module('tcnn', net)
    losses = []

    target_feature = model(style_img)

    loss = TcnnLoss(target_feature)
    model.add_module('loss', loss)

    losses.append(loss)

    return model, losses



def model_tcnn_forward(img, ref,weight_tcnn):

    img = img.reshape(1, nc, imsize, imsize)
    img.requires_grad_()

    tcnn, losses = get_tcnn_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)
   
    with torch.no_grad():
        tcnn(img)

    score = 0
    for sl in losses:
        score += weight_tcnn * sl.loss


    grad = 0


    return score, grad




def model_tcnn(img, ref,weight_tcnn):

    img = img.reshape(1, nc, imsize, imsize)
    img.requires_grad_()

    tcnn, losses = get_tcnn_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)

    tcnn(img)

    score = 0
    for sl in losses:
        score += weight_tcnn * sl.loss


    score.backward()
    grad = img.grad


    return score, grad.flatten()




def model_tcnn_opt(m0,img, ref,weight_tcnn):

    img = img.reshape(1, nc, imsize, imsize)
    img.requires_grad_()

    tcnn, losses = get_tcnn_model_and_losses(
        cnn_normalization_mean, cnn_normalization_std, ref, device)

    tcnn(img)

    score = 0
    for sl in losses:
        score += weight_tcnn * sl.loss

    comp = (score-m0)**2
    
    comp.backward()
    grad = img.grad
    
    return comp, grad

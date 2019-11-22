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

from models import *
from functions import *

from math import exp
import numpy as np
import copy
import cv2
import datetime

load_path = '/home/j263zhou/Desktop/learning/2019_3_Fall/1_image_processing/research/Xinyu_code/data/simoncelli_textures/'
image_tag = 19
hold = 1
direction = 0

def mad_test(imgn, ref_img, hold, direction):

    # load the model
    # model_style, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, ref_img)
    # print('model load success!')
    # print(model_style)

    imgn.data.clamp_(0,1) # transition from [0,255] -> (0,1)?
    input_img = imgn.cpu().detach()
    ref = ref_img.cpu().detach()

    # mad search find the maximal / minimal
    iterations = 500
    for i in range(iterations):
        lamuda = 1 - i * 0.00015
        # ref = ref.to(device)
        # input_img = input_img.to(device)
        #loss1, g1 = model_gram(model_style, input_img.detach(), style_losses)
        loss1, g1 = mse(input_img.detach(), ref.detach())
        
        #loss2, g2 = mse(input_img.detach(), ref.detach())
        #loss2, g2 = model_gram(model_style, input_img.detach(), style_losses)
        loss2, g2 = ssim(input_img.detach(), ref.detach())
        
        # TODO: Need to change the hold
        if hold == 0:
            y, comp = search_grad(ref.detach(), g_2n = g2, g_1n = g1, direction = direction, img = input_img.detach(), mkeep = mse, lamda = lamuda, iterate = i)
        elif hold == 1:
            y, comp = search_grad(ref.detach(), g_2n = g2, g_1n = g1, direction = direction, img = input_img.detach(), mkeep = ssim, lamda = lamuda, iterate = i)
        
        loss_keep, _ = mse(y.detach(), ref.detach())
        loss_change, _ = ssim(y.detach(), ref.detach())
        if i % 10 == 0:
            print('iteration: ' + str(i))
            print('loss1:     ' + str(loss1))
            print('loss2:     ' + str(loss2))
            print('keep:      ' + str( torch.abs(loss_keep - loss1) ) )
            print('change:    ' + str( torch.abs(loss_change - loss2) ) )

            
        input_img = y
        
        #early stop
        if comp < 1e-5:
            print('iteration: ' + str(i))
            print('loss1:     ' + str(loss1))
            print('loss2:     ' + str(loss2))
            print('keep:      ' + str( torch.abs(loss_keep - loss1) ) )
            print('change:    ' + str( torch.abs(loss_change - loss2) ) )
            print('comp: ' + str(comp))
            break

    return input_img

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    #for every imgs
    for imgs in range(1, 2):
        image_tag = imgs

        # load images and add noise
        ref_img = image_loader(load_path + str(image_tag) + ".jpg")
        ref = ref_img * 255
        imgn = gaussian_noise(ref)

        input_img = mad_test(imgn, ref_img, hold = 1, direction = 1)


        imshow(torch.clamp(input_img, 0, 1))


    endtime = datetime.datetime.now()
    print ('total time: ' + str((endtime - starttime).seconds) + 's')

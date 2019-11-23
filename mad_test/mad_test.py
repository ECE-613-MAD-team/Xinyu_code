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
import os

load_path = '/home/j263zhou/Desktop/learning/2019_3_Fall/1_image_processing/research/Xinyu_code/data/simoncelli_textures/'
#image_tag = 19
#hold = 1      # 0: MSE, 1: SSIM
#direction = 0 # 0: add, 1: sub
epi = 1e-6

def img_const(image_tag, noise_tag):

    noise_use = gaussian_noise
    if noise_tag == 0:
        noise_use = gaussian_noise
    elif noise_tag == 1:
        noise_use = blur_noise
    elif noise_tag == 2:
        noise_use = jpeg_noise
    elif noise_tag == 3:
        noise_use = gamma_noise
        
    noise_name = ['gaussian', 'blur', 'jpeg', 'gamma']


    return noise_use, noise_name[noise_tag]

def model_const(hold):

    mkeep = mse
    mkeep_opt = ssim_opt
    mchange = ssim

    if hold == 0:
        mkeep = mse
        mkeep_opt = mse_opt
        mchange = ssim
    elif hold == 1:
        mkeep = ssim
        mkeep_opt = ssim_opt
        mchange = mse

    return mkeep, mkeep_opt, mchange


def mad_test(imgn, ref_img, hold, direction):

    # load the model
    # model_style, style_losses = get_style_model_and_losses(cnn, cnn_normalization_mean, cnn_normalization_std, ref_img)
    # print('model load success!')
    # print(model_style)

    imgn.data.clamp_(0,1) # transition from [0,255] -> (0,1)?
    input_img = imgn
    ref = ref_img
    init_loss = 0
    lamda2 = -1

    # mad search find the maximal / minimal
    iterations = 500
    for i in range(iterations):
        #lamuda = 0.1 - i * 0.00015
        
        mkeep, mkeep_opt, mchange = model_const(hold)
        # ref = ref.to(device)
        # input_img = input_img.to(device)
        #loss1, g1 = model_gram(model_style, input_img.detach(), style_losses)
        loss1, g1 = mkeep(input_img.detach(), ref.detach())
        
        #loss2, g2 = mse(input_img.detach(), ref.detach())
        #loss2, g2 = model_gram(model_style, input_img.detach(), style_losses)
        loss2, g2 = mchange(input_img.detach(), ref.detach())
        lamuda = 0.1

        if i == 0:
            c = lamuda*torch.norm(g2)
            init_loss = loss1
        else:
            pass
        lamuda = c / (torch.norm(g2) ) + 0.02
        #print('iter: ' , i, 'lamuda: ', lamuda, 'c: ', c, 'norm: ', torch.norm(g2))
        
        y, comp, lamda2 = search_grad(ref.detach(), g_2n = g2, g_1n = g1, direction = direction, 
                        img = input_img.detach(), mkeep = mkeep, mkeep_opt = mkeep_opt, 
                        lamda = lamuda, init_loss = init_loss, lamda2 = lamda2)
        
        loss_keep, _ = mkeep(y.detach(), ref.detach())
        loss_change, _ = mchange(y.detach(), ref.detach())
        #print('iter: ', i, 'change: ', loss_change, 'keep', loss_keep, 'comp: ', comp, 'lamda2: ', lamda2)
        if i % 10 == 0:
            print('iteration  : ' + str(i))
            print('keep       : ' + str(loss_keep))
            print('change     : ' + str(loss_change))
            print('keep step  : ' + str( torch.abs(loss_keep - loss1) ) )
            print('change step: ' + str( torch.abs(loss_change - loss2) ) )

            
        input_img = y
        
        #early stop
        if loss_change < 1e-4:
            print('early stop !!!')
            print('iteration  : ' + str(i))
            print('keep       : ' + str(loss_keep))
            print('change     : ' + str(loss_change))
            print('keep step  : ' + str( torch.abs(loss_keep - loss1) ) )
            print('change step: ' + str( torch.abs(loss_change - loss2) ) )
            print('comp       : ' + str(comp))
            break

    return input_img

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    #for every imgs
    for imgs in range(0, 35):
        image_tag = imgs

        # load images and add noise
        ref_img = image_loader(load_path + str(image_tag) + ".jpg")
        ref = ref_img * 255

        # for every kind of noise
        for noise_tag in range(0, 1):
            noise_use, noise_name = img_const(image_tag, noise_tag)

            imgn = noise_use(ref)

            # for every hold and direction
            for h in range(1, 2):
                for d in range(0, 2):
                    input_img = mad_test(imgn, ref_img, hold = h, direction = d)

                    # Save the result
                    result = torch.clamp(input_img, 0, 1)
                    image = result.cpu().clone()  # we clone the tensor to not do changes on it
                    image = image.squeeze(0)      # remove the fake batch dimension
                    image = unloader(image)

                    hold_name = ''
                    if h == 0:
                        hold_name = 'mse'
                    elif h == 1:
                        hold_name = 'ssim'

                    save_dir = './test_result/MSE_vs_SSIM/' + hold_name + '/' 
                    save_file = save_dir + noise_name + '_' + str(image_tag) + '_' + str(d) + '.jpg'
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    image.save(save_file, dpi=(300, 300))


    endtime = datetime.datetime.now()
    print ('total time: ' + str((endtime - starttime).seconds) + 's')

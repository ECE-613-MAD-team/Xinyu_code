import torch


from models import *
from opt import *
from utils import *
from mse import *
from ssim import *
from one_layer import *


#import inspect
#from Pytorch_Memory_Utils import *

import time
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nc = 3
imsize = 256
iterations = 10000



ref_img = image_loader("./data/texture/" + name + ".jpg")
ref_img = ref_img[:,:,0:256,0:256]
#imgn = image_loader("./data/texture/jpeg_10_radish.jpg")



"""

gaussian noise


"""
seed = 999
torch.manual_seed(seed)


if Type == 'noise' :
    imgn = gaussian_noise(float(level),ref_img)
elif Type == 'blur' :
    imgn = gaussian_blur(float(level),ref_img)
else:
    pass




# plt.figure()
# imshow1(imgn, title='distorted image')
# plt.show()




####################################################
imgn.data.clamp_(0,1)
input_img = imgn.detach()
#input_img = torch.load('temp.pt')
ref = ref_img.detach()


iters = 20
prev_loss2 = 0
count = 0
#lamda2 = -0.01
lamda = 0.12
#frame = inspect.currentframe()          # define a frame to track
#gpu_tracker = MemTracker(frame)
start = time.time()


for i in range(iterations):
    """
    compute lamda, loss1 and loss2
    
    """

    
    """
    
    print information
    """
    if i%iters == 0:
        end = time.time()
        print('time per',iters,'iterations',end-start)
        start = end
        print('iteration',i)
        print('lamda',lamda)    
    


    """
    keep model
    """
    #gpu_tracker.track()
    # model which needed keep same
    if int(gd) == 1 or int(gd) == 2:
         loss1, g1 = mse(input_img.detach(), ref.detach())
    else:
         loss1, g1 = model_gram(input_img.detach(), ref.detach())

    if i ==0:
        m0 = loss1
    else:
        pass
    
    if i%iters == 0:
        print('loss1',loss1)
        print('g1',g1.max(),g1.min(),torch.mean(torch.abs(g1)))
    

   
   
    
    """
    
    max/min model
    
    """
    if i > 0:
        prev_loss2 = loss2
    #gpu_tracker.track()
    # min/max this model
    if int(gd) == 1 or int(gd) == 2:
        loss2, g2 = model_gram(input_img.detach(), ref.detach())
    else:
        loss2, g2 = mse(input_img.detach(), ref.detach())

    if i%iters == 0:
       # print('\n')
        print('loss2',loss2)
        print('g2',g2.max(),g2.min(),torch.mean(torch.abs(g2)))

    
        
    """
    
    adjust learning rate
    
    """
    if i == 0:
        fix = lamda*torch.norm(g2) 
        #fix = torch.load('temp_fix.pt')*0.1
    lamda = fix/torch.norm(g2) 
    if torch.abs(prev_loss2-loss2) < 1e-3*torch.abs(loss2):
        lamda = lamda*0.8
    lamda = step_size(lamda0 = lamda, opt = 100, rate1 = 1, rate2 = 1, iteration = i)

    

    
    """
    early stopping
    
    """
    
    if torch.abs(loss2-prev_loss2) < 1e-4*loss2:
        count += 1
    else:
        count = 0
    
    if count > 20:
        print('early stopping!!')
        break


    """
    
    using MAD
    
    """
    if int(gd) == 1 or int(gd) == 2:
        mkeep = mse
    else:
        mkeep = model_gram
    #gpu_tracker.track()
    # change mkeep and xxx_opt in opt.py search_grad function
    y, comp  = search_grad(ref.detach(), 
                                  g = g2, gkeep = g1,
                                  img = input_img.detach(), 
                                  mkeep = mkeep,
                                  init_loss = m0, 
                                  lamda = lamda)
    


    """
    
    print information
    
    """
    if i %iters == 0:
        #print('\n')
        print('cumulate comp:', comp)
        plt.figure()
        imshow(torch.clamp(y,0,1))
        torch.save(input_img,'temp.pt')
        torch.save(fix,'temp_fix.pt')
        print('\n\n')
    if comp > 5:
        print('too big step size, change lamda!!')
        torch.save(input_img,'temp.pt')  
        torch.save(fix,'temp_fix.pt')
        break
    input_img = y

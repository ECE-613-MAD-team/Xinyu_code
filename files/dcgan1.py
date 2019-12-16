from __future__ import print_function # should come fisrt,cause __future__ module change foundation things 
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


from data_io import *
import time, datetime
import json


# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
# random.randint:   [low, high) 'uniform discrete distribution'
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)   # seed the random number generator for both CPU and CUDA

# Root directory for dataset
dataroot = "data/celeba/"

# out dir for images_samples / parameters / training models&output results
outroot = './out'
now = datetime.datetime.now()
subname = os.path.join(outroot,now.strftime("%Y_%m_%d"))


if not os.path.exists(subname):
  os.makedirs(subname)


parser = argparse.ArgumentParser()   # class use to deal with argument(add, return)
parser.add_argument("--workers",type = int,help = "",default = 12)
parser.add_argument("--batch_size",type = int,help = "",default = 32)

parser.add_argument("--nc", type = int,help = "",default = 3)


"""

image_size = (64*zx)*(64*zx)

"""
# parser.add_argument("--image_size", type = int,help = "",default = 256)
parser.add_argument("--nz",type = int,help = "",default = 100)
parser.add_argument("--zx",type = int,help = "",default = 1)

parser.add_argument("--ngf",type = int,help = "",default = 64)
parser.add_argument("--ndf",type = int,help = "",default = 64)
parser.add_argument("--num_epochs",type = int,help = "",default = "100")

parser.add_argument("--lr", type = float, help = "",default = 0.0002)
parser.add_argument("--beta1", type = float, help = "",default = 0.5)


parser.add_argument("--device", type = str, help = "",default = 'cpu')

args = parser.parse_args()


###############################
workers = args.workers
batch_size = args.batch_size
nc = args.nc


nz = args.nz
zx = args.zx

"""
64 because net structure

6 layer conv with :

kernel size=5   stride=2   padding = 2   outpadding = 1

"""
image_size = 64*zx #args.image_size

ngf = args.ngf
ndf = args.ndf

num_epochs = args.num_epochs
beta1 = args.beta1
lr = args.lr


device = torch.device(args.device)
######################################


"""

back up parameters

"""
pdict = vars(args)
print(pdict)
with open(os.path.join(subname, 'params.json'), 'w') as f:
    flags_dict = {k:pdict[k] for k in pdict}
    json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0,bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size.  
            #nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1,bias=False),
            #nn.BatchNorm2d(ngf * 8),
            #nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 4, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf * 2, ngf , 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size
        )

    def forward(self, input):
        return self.main(input)




class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            #nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(ndf * 16),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)




def main():


  # data preparation
  # reture numpy array 
  #data_iter   = get_texture_iter(dataroot, npx=image_size, mirror=False, batch_size=batch_size)
  #print("data type:", type(data_iter))
  #print("data shape", data_iter.shape)
  #plt.figure(figsize = (8,8))
  #plt.axis("off")
  #plt.title("training images")
  # 64*3*64*64 -> 3*530*530   #TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  #data = torch.from_numpy(data_iter).float().to('cpu')
  #plt.imshow(np.transpose(vutils.make_grid(data[:batch_size], padding=2, normalize=True),(1,2,0)))
  #plt.show()
  
  
  
  dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),   #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    # (image - mean)/ std
                           ]))
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers= workers)
  data_iter = next(iter(dataloader))[0]
  print("data type:", type(data_iter))
  print("data shape", data_iter.shape)
  plt.figure(figsize = (8,8))
  plt.axis("off")
  plt.title("training images")
  # 64*3*64*64 -> 3*530*530   #TypeError: can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
  #data = torch.from_numpy(data_iter).float().to('cpu')
  plt.imshow(np.transpose(vutils.make_grid(data_iter[:batch_size], padding=2, normalize=True),(1,2,0)))
  plt.show()
  
  #device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
  print('device:',device)




  # Create the generator
  netG = Generator().to(device)

  # # Handle multi-gpu if desired
  # if (device.type == 'cuda') and (ngpu > 1):
    # netG = nn.DataParallel(netG, list(range(ngpu)))

  # Apply the weights_init function to randomly initialize all weights
  #  to mean=0, stdev=0.2.
  netG.apply(weights_init)

  # Print the model
  print(netG)



  # Create the Discriminator
  netD = Discriminator().to(device)

  # # Handle multi-gpu if desired
  # if (device.type == 'cuda') and (ngpu > 1):
    # netD = nn.DataParallel(netD, list(range(ngpu)))

  # Apply the weights_init function to randomly initialize all weights
  #  to mean=0, stdev=0.2.
  netD.apply(weights_init)

  # Print the model
  print(netD)




  # Initialize BCELoss function
  criterion = nn.BCELoss()

  # Create batch of latent vectors that we will use to visualize
  #  the progression of the generator
  fixed_noise = torch.randn(batch_size, nz, zx, zx, device=device)
  
  # Establish convention for real and fake labels during training
  real_label = 1
  fake_label = 0

  # Setup Adam optimizers for both G and D
  optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
  optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))



  # Training Loop

  # Lists to keep track of progress
  img_list = []
  BG_losses = []
  BD_losses = []
  BD_x = []
  BD_G_z1 = []
  BD_G_z2 = []


  iters = 0

  print("Starting Training Loop...")

  # For each epoch
  for epoch in range(num_epochs):
    # For each batch in the dataloader
    timed1 = time.time()
    #for i in range(500):
    for i,data in enumerate(dataloader,0):
        #data_iter   = get_texture_iter(dataroot, npx=image_size, mirror=False, batch_size=batch_size)
        #data = torch.from_numpy(data_iter).float().to(device)
        
        if i == 0:
            print("netD input shape:", data[0].shape,'\n')

        timed2 = time.time()
        #print('data load:',timed2-timed1)


        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        timetd1 = time.time()
        netD.zero_grad()
        # Format batch
        #real_cpu = data.to(device)
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        
        
        """
        
        batch_size*zx*zx is output element numbers
        shape should be like B*zx*zx
        
        """
        label = torch.full((b_size*zx*zx,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)

        if i == 0:
            print("netD output shape", output.shape,'\n')


        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        """
        
        input noise shape B*nz*zx*zx
        
        """
        noise = torch.randn(b_size, nz, zx, zx, device=device)
        # Generate fake image batch with G
        
        
        
        if i == 0:
            print("netG input shape:", noise.shape,'\n')
        """
        
        fake image size : B*nc*(64*zx)*(64*zx)
        
        """
        fake = netG(noise)

        if i == 0:
            print("netG output shape:",fake.shape,'\n')
        #print("conv1 weight shape:", netG.main[0].weight.shape)

        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()
        timetd2 = time.time()
        #print('netd train:',timetd2-timetd1)

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        timetg1 = time.time()
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        timetg2 = time.time()
        #print('netg train:',timetg2-timetg1)

        timec1 = time.time()
        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        BG_losses.append(errG.item())
        BD_losses.append(errD.item())
        BD_x.append(D_x)
        BD_G_z1.append(D_G_z1)
        BD_G_z2.append(D_G_z2)

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 100 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        timec2 = time.time()
        #print('collect time:',timec2-timec1)
        timed1 = time.time()
    if epoch % 1 == 0:
      torch.save({
            'epoch': epoch,
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'img': img_list,
            'lossG':BG_losses,
            'lossD':BD_losses,
            'D_x':BD_x,
            'D_G_z1':BD_G_z1,
            'D_G_z2':BD_G_z2
            
            }, os.path.join(subname,'model.tar'))





if __name__ == '__main__':
  main()

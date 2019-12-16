import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

outroot = './out'
subname = os.path.join(outroot,'2019_10_26')

"""
'epoch' 'netG_state_dict' 'netD_state_dict'

'optimizerG_state_dict' 'optimizerD_state_dict'

'img' 'lossG' 'lossD' 'D_x' 'D_G_z2' 'D_G_z1'

"""


# netG = netG = Generator(ngpu).to(device)
# netD = Discriminator(ngpu).to(device)



checkpoint = torch.load(os.path.join(subname,'model.tar'), map_location = 'cpu')

#print('epoch:',checkpoint['epoch'],'\n\n')

# Print model's state_dict
#print("netG's state_dict:")
#for param_tensor in checkpoint['netG_state_dict']:
#    print(param_tensor, "\t", checkpoint['netG_state_dict'][param_tensor].size())

#print('\n\n')

#print("netD's state_dict:")
#for param_tensor in checkpoint['netD_state_dict']:
#    print(param_tensor, "\t", checkpoint['netD_state_dict'][param_tensor].size())

# # Print optimizer's state_dict
# print("OptimizerG's state_dict:")
# for var_name in checkpoint['optimizerG_state_dict']:
    # print(var_name, "\t", checkpoint['optimizerG_state_dict'][var_name])

# print("OptimizerD's state_dict:")
# for var_name in checkpoint['optimizerD_state_dict']:
    # print(var_name, "\t",(checkpoint['optimizerD_state_dict'][var_name])


# visulize fake images build upon fixed noise

# print((checkpoint['img'][1]).size())
# plt.imshow(np.transpose(checkpoint['img'][20],(1,2,0)))
# plt.show()


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(checkpoint['lossG'],label="G")
plt.plot(checkpoint['lossD'],label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('plot.png', dpi=300, bbox_inches='tight')


plt.show()
#plt.savefig('train.jpg')

#print(checkpoint['img'][8].shape)  # 3*530*530
ims = 256
index = 40

plt.figure()
print(type(checkpoint['img'][index]))
img = np.array(np.transpose(checkpoint['img'][index],(1,2,0)))
plt.imshow(img)
plt.show()
print(type(img),img.shape)
plt.imsave('00.jpg',img,dpi=300)

plt.figure()
img_1 = np.array(np.transpose(checkpoint['img'][index][:,2+ims:2+2*ims,2:2+ims],(1,2,0)))
plt.imshow(img_1)
plt.show()
print(type(img_1),img_1.shape)
plt.imsave('01.jpg',img,dpi=300)






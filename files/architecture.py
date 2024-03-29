##################################################################################################################

PYTORCH DCGAN

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

##################################################################################################################

6 LAYER SGAN

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 16, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size.  
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf * 2, ngf , 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf, nc, 5, 2, 2, 1, bias=False),
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
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 5, 2, 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

#########################################################################################################


5 LAYER SGAN

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 8, ngf * 4, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf * 2, ngf , 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # state size
            nn.ConvTranspose2d( ngf, nc, 5, 2, 2, 1, bias=False),
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
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 5, 2, 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

######################################################################################################################

4 LAYER SGAN

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 5, 2, 2, 1,bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. 
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf * 2, ngf , 5, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size
            nn.ConvTranspose2d( ngf, nc, 5, 2, 2, 1, bias=False),
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
            nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 5, 2, 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)




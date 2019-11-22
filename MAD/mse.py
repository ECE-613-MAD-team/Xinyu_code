import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imsize = 256
nc = 3

weight_mse = 2e4


def mse(img, ref):
    img = img.reshape(1, nc, imsize, imsize)

    img = img.to(device)
    ref = ref.to(device)

    img.requires_grad_()

    N = nc * imsize * imsize
    loss = weight_mse * ((img - ref) ** 2).sum() / (N)
    loss.backward()
    #
    #    del ref
    #    torch.cuda.empty_cache()
    return loss.cpu(), img.grad.flatten().cpu()


def mse_opt(m0, temp, ref):
    temp = temp.reshape(1, nc, imsize, imsize)

    m0 = m0.to(device)
    temp = temp.to(device)
    ref = ref.to(device)

    temp.requires_grad_()

    N = nc * imsize * imsize
    loss_mse = weight_mse * ((temp - ref) ** 2).sum() / (N)
    comp = (m0 - loss_mse) ** 2
    #print('comp',comp,m0-loss_mse,m0,loss_mse)
    comp.backward()

    return comp.cpu(), temp.grad.cpu()
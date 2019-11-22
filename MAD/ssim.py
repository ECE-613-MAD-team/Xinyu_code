import torch
import pytorch_ssim



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_ssim = 2e3
nc = 3
imsize = 256


def ssim(img, ref):
    img = img.reshape(1, nc, imsize, imsize)

    img = img.to(device)
    ref = ref.to(device)

    img.requires_grad_()

    ssim_value = pytorch_ssim.ssim(ref, img)
    ssim_loss = pytorch_ssim.SSIM()
    ssim_out = -weight_ssim * ssim_loss(ref, img)
    ssim_out.backward()

    #    del ref
    #    torch.cuda.empty_cache()
    return ssim_value.cpu(), img.grad.flatten().cpu()





def ssim_opt(m0, temp, ref):
    temp = temp.reshape(1, nc, imsize, imsize)

    m0 = m0.to(device)
    temp = temp.to(device)
    ref = ref.to(device)

    # _, nc, imsize, imsize = temp.shape
    temp.requires_grad_()

    ssim_value = pytorch_ssim.ssim(ref, temp)
    ssim_loss = pytorch_ssim.SSIM()
    ssim_out = -weight_ssim * ssim_loss(ref, temp)
    comp = ((-weight_ssim * m0) - ssim_out) ** 2
    comp.backward()

    return comp.cpu(), temp.grad.cpu()

import torch
import numpy as np





"""

Adam


"""

def Adam(m0, xm, ref, mkeep_opt):
    
    
    lr = 0.0001
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8

    theta_0 = 0
    m_t = 0 
    v_t = 0 
    t = 0
    
    var = 1
    while var == 1:
        t += 1
        #print('t',t)
        comp, g_t = mkeep_opt(m0,xm.detach(),ref.detach())
        m_t = beta_1*m_t + (1-beta_1)*g_t     # consider 90% of previous, and 10% of current
        v_t = beta_2*v_t + (1-beta_2)*(g_t*g_t) # 99.9% of previous (square grad), and 1% of current
        m_cap = m_t/(1-(beta_1**t))      #calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))      #calculates the bias-corrected estimates
        
        xm_prev = xm
        xm = xm - (lr*m_cap)/(torch.sqrt(v_cap)+epsilon)
        if comp < 1e-10:    #checks if it is converged or not
            break
            
    return comp, xm 

"""
g: gradient at xm

"""

def bisection(f, lower, upper, g, ref, x, xm):

    obj, _ = f(x, ref)
    var = 1
    a = lower
    b = upper
    m = (a+b)/2
    while var == 1:
        if (f(xm+a*g,ref)[0]-obj)*(f(xm+m*g,ref)[0]-obj) < 0:
            b = m
            m = (a+b)/2
        elif (f(xm+m*g,ref)[0]-obj)*(f(xm+b*g,ref)[0]-obj) < 0:
            a = m
            m = (a+b)/2
        else:
            #print('a wider bound!')
            #print('temp comp:',f(xm+m*g,ref)[0]-obj )
            break
         
        if b-a < 1e-6:
             break
    comp = f(xm+m*g,ref)[0]-obj
    return comp, (xm + m*g)




def search_grad(ref, g, gkeep, img = None, mkeep = None, mkeep_opt = None, lamda = None):
    
    _,nc,imsize,_ = img.shape
    r = 1
    step = 0.001
#     r = 1
#     step = 0.01
    N = 2*r/step
    vsearch = np.linspace(-1.5*r,1.5*r,N)
    # _, nc, _, _ = x.shape 
    
    
  
    #####   project
    gm = g - torch.mul(torch.div(torch.dot(g,gkeep), torch.dot(gkeep,gkeep)) ,gkeep)
    
   # plt.hist(gm.cpu(), 1000)
   #plt.show()
    #print('gm',gm.max(),gm.min(),torch.mean(torch.abs(gm)))
    
    
    ################# 
    xm = torch.sub(img.flatten(), torch.mul(lamda, gm)).reshape(1,nc,imsize,imsize)
    #xm = torch.add(img.flatten(), torch.mul(lamda, gm)).reshape(1,nc,imsize,imsize)
    
    #print('xm-img:', (xm-img).sum())
   
    # y = xm
    
    
# #     ####################################
    # m0, _ = mkeep(img.detach(),ref.detach())
# #   #m0,_ = mkeep(model_style, img.detach(), style_losses)
  
    # #comp, y = Adam(m0.detach(),xm.detach(),ref.detach(),mkeep_opt = mkeep_opt)
    
    gn = mkeep(xm.detach(), ref.detach())[1].reshape(1,nc,imsize,imsize)
    comp, y = bisection(mkeep, -0.5, 1, gn, ref, img, xm)
    # m1, gn = mkeep(xm.detach(), ref.detach())
    # #m1, gn = mkeep(model_style, xm.detach(), style_losses)
    
   # # print('gn',gn.max(),gn.min(),torch.mean(torch.abs(gn)))
    # comp = torch.abs(m1-m0)
   # # print('comp',comp)
    
    # for i,v in enumerate(vsearch):
        # # print('v:',v)
        # temp_im = xm.flatten() + v*gn
        # temp_im = temp_im.reshape(1,nc,imsize,imsize)
        # #print('temp_im-xm:', (temp_im-xm).sum())
        # temp_mkeep, _ = mkeep(temp_im.detach(), ref.detach())
        # #temp_mkeep, _ = mkeep(model_style, temp_im.detach(), style_losses)
        # temp_comp =  torch.abs(temp_mkeep-m0)
        # #if i%1000 == 0:
        # #    print('v temp_comp',v,temp_comp)
        # if temp_comp  < comp:
            # #print('!',v)
            # comp = temp_comp
            # y = temp_im
            # if temp_comp < 0.001:
                # break
   # print('y-img',(y-img).sum())        
    return y, comp

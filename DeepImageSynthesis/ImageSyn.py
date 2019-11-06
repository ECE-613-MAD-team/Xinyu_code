import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from .Misc import *

def mpool5(net, constraints, init=None, bounds=None, callback=None, minimize_options=None, gradient_free_region=None):
    '''
    This function generates the image by performing gradient descent on the pixels to match the constraints.

    :param net: caffe.Classifier object that defines the network used to generate the image
    :param constraints: dictionary object that contains the constraints on each layer used for the image generation
    :param init: the initial image to start the gradient descent from. Defaults to gaussian white noise
    :param bounds: the optimisation bounds passed to the optimiser
    :param callback: the callback function passed to the optimiser
    :param minimize_options: the options passed to the optimiser
    :param gradient_free_region: a binary mask that defines all pixels that should be ignored in the in the gradient descent   
    :return: result object from the L-BFGS optimisation
    '''

    #if init==None:
    #    init = np.random.randn(*net.blobs['data'].data.shape)
    #print('init:', init.max(), init.min())
    
     #get indices for gradient
    layers, indices = get_indices(net, constraints)
    
    #function to minimise 
    def f(x):
        x = x.reshape(*net.blobs['data'].data.shape)
        net.forward(data=x, end=list(layers)[min(len(layers)-1, indices[0]+1)])
        f_val = 0
        #clear gradient in all layers
        for index in indices:
            net.blobs[list(layers)[index]].diff[...] = np.zeros_like(net.blobs[list(layers)[index]].diff)
                
        for i,index in enumerate(indices):
            layer = list(layers)[index]
            for l,loss_function in enumerate(constraints[layer].loss_functions):
                constraints[layer].parameter_lists[l].update({'activations': net.blobs[layer].data.copy()})
                val, grad = loss_function(**constraints[layer].parameter_lists[l])
                f_val += val
                net.blobs[layer].diff[:] += grad
            #gradient wrt inactive units is 0
            net.blobs[layer].diff[(net.blobs[layer].data == 0)] = 0.
            if index == indices[-1]:
                f_grad = net.backward(start=layer)['data'].copy()
            else:        
                net.backward(start=layer, end=list(layers)[indices[i+1]])                    

        if gradient_free_region!=None:
            f_grad[gradient_free_region==1] = 0    
        #print(f_grad.shape,f_grad.max(),f_grad.min())
        return [f_val, np.array(f_grad.ravel(), dtype=float)]  
             
    #f(init)  
    loss, f_grad = f(init)
    #print(f_grad.shape,f_grad.max(),f_grad.min(),np.mean(f_grad))
    #result = minimize(f, init,
    #                      method='L-BFGS-B', 
    #                      jac=True,
    #                      bounds=bounds,
    #                      callback=callback,
    #                      options=minimize_options)
    #return  result
    return loss, f_grad
        


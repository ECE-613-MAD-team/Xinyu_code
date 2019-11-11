# work has been done

Using MAD make comparision between VGG+Gram matrix developed by Gatys et al. and MSE. Code maily in Example.ipynb in which preprocessing, models and MAD are implemented. You need caffe and pytorch to run it.


- initial image gaussian noise polluted image

a1.jpg - Best image for Gatys network. Hold MSE

a2.jpg - Worst ...      Gatys ...

a3.jpg - initial image

a4/a5 Best MSE/Worst MSE.Hold Gatys network.

    initial: pebbles.jpg
          a1
           | 
    a4 -- a3 -- a5
           |
          a2

-initial image generate by sgan4 network.(paper:Texture Synthesis with Spatial Generative Adversarial Networks)

b1 - b5 similiar to a1 - a5 but using a different initial image which is b3.jpg.



# What's next
need more reference images (apart from pebbles) 

need more models(paper: WHAT DOES IT TAKE TO GENERATE NATURAL TEXTURES, SSIM, ...)

need more initial images(apart form gaussian noise and image generate by sgan4)


Maybe design a new quality metric.

Maybe utilize some interesting points get from MAD comparison.

# Models going to compare:

* MSE
* SSIM
* WHAT DOES IT TAKE TO GENERATE NATURAL TEXTURES (gram-matrix + one-layer-net)
* gram-matrix + VGG (conv5)
* Different layers? (Maybe)



# Mdels comparison results:

### Gram-matrix + VGG VS MSE


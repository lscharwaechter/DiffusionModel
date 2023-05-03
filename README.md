# Diffusion Model

In this project, a diffusion model is applied to artificially generate new images from a given dataset. The diffusion process is inspired by [[1]](#1) and [[2]](#2) and the code is based on the implementation of [this notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL).

The model consists of a simplified UNet architecture with several convolutional blocks and residual connections. Instead of a full diffusion process, each image is assigned a random time step value that is passed to the model together with the image. A Positional Encoding layer embeds the time step into a vector of sines and cosines (as in the Transformer architecture). During training, a batch of images is transformed into its noisy version using the assigned diffusion time stamp. The model then learns to predict the added noise levels of the input batch. The loss function compares the predicted noise levels with the true noise levels, while the gradients are used to minimize this difference. This strategy is computationally more effective than passing each image into a full diffusion process involving all time steps. During inference however, the model is given a random noise Tensor which runs through the whole diffusion process reversely: Starting with the last time step, the noise Tensor is passed as an image to the model and the predicted noise level is then substracted from the image. This is done interatively unitl the first timestep is reached, which represents a generated, noiseless image. The amount of noise added/reduced to the image is predefined by the betas vector and determines a linearly increasing amount of noise per time step. 

As an exemplary dataset, an image dataset for crater detection on Mars and Moon surface is used [[3]](#3). The training set consists of $98$ images, which are resized to a dimensionalty of $64 \times 64$, normalized to a range of $[-1,1]$.

## References
<a id="1">[1]</a>
Yale Song, David Demirdjian, and Randall Davis (2011).
Tracking Body and Hands For Gesture Recognition: NATOPS Aircraft Handling Signals Database.
In Proceedings of the 9th IEEE International Conference on Automatic Face and Gesture Recognition.

<a id="2">[2]</a> 
R. Guidotti et al. (2020). 
Explaining Any Time Series Classifier.
IEEE Second International Conference on Cognitive Machine Intelligence (CogMI)


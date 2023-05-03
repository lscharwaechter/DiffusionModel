# Diffusion Model

In this project, a diffusion model is applied to artificially generate new images from a given dataset. The diffusion process is inspired by [[1]](#1) and [[2]](#2) and the code is based on the implementation of [this notebook](https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL).

The model consists of a simplified UNet architecture with several convolutional blocks and residual connections. Instead of a full diffusion process, each image is assigned a random time step value that is passed to the model together with the image. A Positional Encoding layer embeds the time step into a vector of sines and cosines (as in the Transformer architecture [[3]](#3)). During training, a batch of images is transformed into its noisy version, where the degree of noise is estimated using the assigned diffusion time stamp. The model then learns to predict the added noise levels of the input batch. The loss function compares the predicted noise levels with the true noise levels, while the gradients are used to minimize this difference. This strategy is computationally more effective than passing each image into a full diffusion process involving all time steps. During inference however, the model is given a random noise Tensor which runs through the whole diffusion process reversely: Starting with the last time step, the noise Tensor is passed as an image to the model and the predicted noise level is then substracted from the image. This is done interatively unitl the first timestep is reached, which represents a generated, noiseless image. The amount of noise added/reduced to the image is predefined by the betas vector and determines a linearly increasing amount of noise per time step. 

As an exemplary dataset, an image dataset for crater detection on Mars and Moon surface is used [[4]](#4). The training set consists of $98$ images, which are resized to a dimensionalty of $64 \times 64$, normalized to a value range of $[-1,1]$.

<p align="center">
<img src="https://user-images.githubusercontent.com/56418155/235883817-275e5a76-12f1-4a4d-8307-c834f22c243f.png" alt="Overview_new" width="70%">
</p>

The above images are generated after $500$ epochs with a batch size of $16$. Rough structures can be recognized, but they still lack sharp contours. A higher model complexity or number of epochs may improve the current outcome.

## References
<a id="1">[1]</a>
J. Ho et al. (2020), "Denoising Diffusion Probabilistic Models",
34 Conference on Neural Information Processing Systems (NeurIPS 2020),
Available: https://arxiv.org/abs/2006.11239

<a id="2">[2]</a> 
Dhariwal and Nichol (2021), "Diffusion Models Beat GANs on Image Synthesis",
Advances in Neural Information Processing Systems 34 (NeurIPS 2021),
Available: https://arxiv.org/abs/2105.05233

<a id="3">[3]</a> 
A. Vaswani et al. (2017), “Attention is all you need”,
Advances in Neural Information Processing Systems 30 (NeurIPS 2017),
Available: https://arxiv.org/abs/1706.03762

<a id="4">[4]</a> 
https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset

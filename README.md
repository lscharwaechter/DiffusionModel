# Text-conditioned Diffusion Model
In this project, a multi-modal text-conditioned image diffusion model is trained to generate realistic crater images from short prompts like "mars" or "moon". The core idea builds upon the Denoising Diffusion Probabilistic Model (DDPM) framework introduced in [[1]](#1), leveraging a reverse denoising process to generate high-quality images from pure noise [[2]](#2).

A conditional U-Net architecture processes noisy images along with a timestep embedding and a pretrained text embedding. The timestep and text features are injected into each residual block of the U-Net. The timestep is used to only predict the added noise at a certain level and immediately reconstruct the original image during training from that noise level, which is computationally more effective than passing each image into a full diffusion process involving all time steps (following the core DDPM approach). A DistilBERT language model is utilized to encode text prompts into a latent vector space. This vector is appended to the diffusion process and allows dynamic image generation from simple words like "mars" or even more elaborate phrases (future work). At inference, the model samples a random vector from a Gaussian Noise distribution and denoises it iteratively from T=500 to 0, using its learned noise predictions to recover a clean image step by step. The amount of noise added/reduced to the image is predefined by the betas vector $\beta_t$ and determines a linearly increasing amount of noise per time step. 

<b>Number of channels in th U-Net:</b>
- Input:	3 (RGB)
- Encoder:	64 -> 128 -> 256
- Decoder:	256 -> 128 -> 64
- Output: 3 (RGB)

<p align="center">
<img src="https://github.com/user-attachments/assets/a34df97b-b39c-4dc4-a725-af837846601f" alt="Overview_new" width="73%">
</p>

As an exemplary dataset, an image dataset for crater detection on Mars and Moon surface is used [[3]](#3). It consists of $142$ images in total, which are resized to a dimensionality of $64 \times 64$ and normalized to a value range of $[-1,1]$. The dataset is imbalanced, with roughly 5x more Mars images than Moon images. This was handled using a weighted sampling strategy to ensure equal representation during training. After 1000 epochs, the model generates new images by prompting either "moon" (left) or "mars" (right): 

<p align="center">
<img src="https://github.com/user-attachments/assets/d9ff2a3a-41f7-404f-8e5a-aa4daf96b557" alt="Overview_new" width="50%">
</p>

With only 142 total images, this project demonstrates that meaningful generative performance can be achieved even in small, highly specialized domains, especially when conditioning on compact semantic prompts. 

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
https://www.kaggle.com/datasets/lincolnzh/martianlunar-crater-detection-dataset

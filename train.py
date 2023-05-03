# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from model import SimpleUnet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 500
LR = 0.001

def beta_schedule(timesteps, start=0.0001, end=0.02):
    '''
    Returns a schedule of the betas for a given amount of timesteps.
    In this example a linear scheduler is used.
    '''
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device=device):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean, variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = beta_schedule(timesteps=T)

# Pre-calculate different terms for the closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

#%%

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

# Load the dataset from the "craters" folder
dataset = ImageFolder('craters', transform=transform)

# Define the data loader with a batch size of 16
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

#%%

# Plot exemplary forward diffusion
image = next(iter(dataloader))[0]
plt.figure(figsize=(15,15))
num_images = 10
stepsize = int(T/num_images)

def show_image(image: torch.Tensor):
    # In case there is a batch dimension, take the first image
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    image = (image+1)/2
    image = torch.clamp(image,min=0,max=1).permute(1,2,0)
    plt.imshow(image) 

for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img, noise = forward_diffusion_sample(image, t)
    show_image(img)
    plt.axis('off')
    
#%%

# Plot existing images of the dataset
plt.figure(figsize=(15,15))
for idx in range(0, T, stepsize):
    t = torch.Tensor([idx]).type(torch.int64)
    plt.subplot(1, num_images+1, int(idx/stepsize) + 1)
    img = next(iter(dataloader))[0]
    show_image(img)
    plt.axis('off')

#%%

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (substract the noise prediction from the current image
    # with the beta weighting) to get the pixel mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    # get the predefined noise variance for the given timestep
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    # The first timestep should be noiseless
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

@torch.no_grad()
def sample_plot_image():
    '''
    Samples a noise vector and iteratively reduces the noise every
    timestep to generate an image 
    '''
    
    img_size = IMG_SIZE
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)

    for i in reversed(range(0,T)):
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_image(img.detach().cpu())
    plt.show()            

#%%

def loss_L1(model, x_0, t):
    '''
    This function estimates the predicted noise level of input x_0 at timestep
    t, compares the prediction with the true added noise and returns
    the L1 loss between both levels.
    '''
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

#%%

model = SimpleUnet()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
      optimizer.zero_grad()

      t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
      
      if batch[0].shape[0] == BATCH_SIZE:
          loss = loss_L1(model, batch[0], t)
          loss.backward()
          optimizer.step()

      if epoch % 5 == 0 and step == 0:
        print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")
        sample_plot_image()

torch.save(model.state_dict(), "diffusion_model.pth")
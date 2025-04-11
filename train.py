import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms, utils
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from models import UNet

# Load Dataset and Transform
data_dir = str('/craters')
'''
Needs to have the following folder structure:
    /craters
        /mars
        /moon
'''

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = ImageFolder(data_dir, transform=transform)
num_classes = len(dataset.classes)

# Handle imbalance with weighted sampling
class_counts = [0] * num_classes
for _, label in dataset.samples:
    class_counts[label] += 1
class_weights = [1.0 / count for count in class_counts]
sample_weights = [class_weights[label] for _, label in dataset.samples]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)

# Load DistilBERT Text Encoder
tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
text_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased")
raw_text_dim = 768
text_emb_dim = 256

# Parameters for the Diffusion Process
T = 500
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
alphas = 1. - betas
alpha_hats = torch.cumprod(alphas, dim=0)

# Sample random images from noise using prompts
@torch.no_grad()
def sample(model, device, prompt, n=16):
    model.eval()
    x = torch.randn(n, 3, 64, 64).to(device)
    text_inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    text_embedding = text_model(**text_inputs).last_hidden_state[:, 0, :].repeat(n, 1)

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t / T], device=device).repeat(n, 1)
        pred_noise = model(x, t_tensor, text_embedding)
        beta_t = betas[t].to(device)
        alpha_t = alphas[t].to(device)
        alpha_hat_t = alpha_hats[t].to(device)
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        x = (1 / alpha_t.sqrt()) * (x - ((1 - alpha_t) / (1 - alpha_hat_t).sqrt()) * pred_noise) + beta_t.sqrt() * noise
    return x

# Visualize the sampled images
def plot_samples(samples, title=""):
    samples = samples.cpu().clamp(-1, 1)
    grid = utils.make_grid(samples, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.title(title)
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

# Training
def train(model, dataloader, optimizer, device, epochs):
    model.train()
    for epoch in range(epochs):
        for x, labels in dataloader:
            x = x.to(device)
            prompts = [dataset.classes[label] for label in labels]
            text_inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embedding = text_model(**text_inputs).last_hidden_state[:, 0, :]

            t = torch.randint(0, T, (x.size(0),), device=device).long()
            noise = torch.randn_like(x)
            alpha_hat_t = alpha_hats.to(device)[t].view(-1, 1, 1, 1)
            noisy_x = (alpha_hat_t.sqrt() * x + (1 - alpha_hat_t).sqrt() * noise)

            pred_noise = model(noisy_x, t.float().unsqueeze(-1) / T, text_embedding)
            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss.item():.4f}")

        if (epoch + 1) % 50 == 0:
            print(f"\n[Epoch {epoch + 1}] Sampling 'mars'...")
            mars_samples = sample(model, device, prompt="mars")
            plot_samples(mars_samples, title=f"Mars - Epoch {epoch + 1}")

            print(f"\n[Epoch {epoch + 1}] Sampling 'moon'...")
            moon_samples = sample(model, device, prompt="moon")
            plot_samples(moon_samples, title=f"Moon - Epoch {epoch + 1}")

# Run Training
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_model = text_model.to(device)
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, dataloader, optimizer, device, epochs=1000)
    # Save model
    torch.save(model.state_dict(), "text_crater_diffusion.pt")

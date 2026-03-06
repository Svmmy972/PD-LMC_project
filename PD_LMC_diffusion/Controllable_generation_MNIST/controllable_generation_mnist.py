import os
import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMPipeline
from tqdm.auto import tqdm


# ==========================================================
# 0) Utils
# ==========================================================
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==========================================================
# 1) Data
# ==========================================================
def get_mnist_dataloaders(
    data_dir="./data",
    batch_size=256,
    num_workers=2,
):
    """
    Returns MNIST train/test dataloaders in [-1, 1] range
    so that the classifier works directly on DDPM outputs.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader



# ==========================================================
# 2) Classifier
# ==========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)   
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)   
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def evaluate_classifier(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            logits = model(imgs)
            loss = F.cross_entropy(logits, lbls)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(dim=1) == lbls).sum().item()
            total += imgs.size(0)

    return total_loss / total, total_correct / total


def train_classifier(
    model,
    train_loader,
    test_loader,
    device,
    epochs=3,
    lr=2e-3,
    save_path=None,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = F.cross_entropy(logits, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (logits.argmax(dim=1) == lbls).sum().item()
            total += imgs.size(0)

        train_loss = running_loss / total
        train_acc = running_correct / total
        test_loss, test_acc = evaluate_classifier(model, test_loader, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return history


def plot_classifier_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(history["train_acc"], label="train acc")
    axs[0].plot(history["test_acc"], label="test acc")
    axs[0].set_title("Classifier accuracy")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(history["train_loss"], label="train loss")
    axs[1].plot(history["test_loss"], label="test loss")
    axs[1].set_title("Classifier loss")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def load_classifier(
    checkpoint_path,
    device,
):
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model



# ==========================================================
# 3) DDPM
# ==========================================================
def load_ddpm_pipeline(
    model_id="1aurent/ddpm-mnist",
    device="cpu",
):
    pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    return pipe, pipe.unet, pipe.scheduler


def make_initial_noise(
    seed,
    shape=(1, 1, 28, 28),
    device="cpu",
):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape, generator=g, device=device)


def x0_hat_from_eps(x_t, noise_pred, alpha_bar_t):
    return (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)


def tensor_to_image(x):
    return (x / 2 + 0.5).clamp(0, 1).squeeze().detach().cpu().numpy()


@torch.no_grad()
def classifier_probs(classifier, x):
    logits = classifier(x)
    return F.softmax(logits, dim=-1)


# ==========================================================
# 4) Guided / non-guided generation
# ==========================================================
@dataclass
class GenerationResult:
    img_tensor: torch.Tensor
    img_numpy: np.ndarray
    lambda_history: list
    prob_history: list
    target_digit: int
    seed: int
    guided: bool


def generate_digit(
    unet,
    scheduler,
    classifier,
    device,
    target_digit=4,
    use_pdlmc=True,
    lr_lambda=0.01,
    lr_x=2.0,
    seed=42,
    num_steps=1000,
    target_confidence=0.9,
    grad_clip=0.05,
):
    """
    Single-image generation from a fixed seed.
    Guided and non-guided can be compared fairly by using the same seed.
    """
    initial_noise = make_initial_noise(seed=seed, shape=(1, 1, 28, 28), device=device)
    x_t = initial_noise.clone()

    lmbda = torch.tensor(0.0, device=device)
    lambda_history = []
    prob_history = []

    scheduler.set_timesteps(num_steps)

    for t in tqdm(
        scheduler.timesteps,
        desc=f"Generation target={target_digit} guided={use_pdlmc}",
        leave=False
    ):
        if use_pdlmc:
            # keep graph through UNet for the guided branch
            x_t = x_t.detach().requires_grad_(True)
            noise_pred = unet(x_t, t).sample

            alpha_bar_t = scheduler.alphas_cumprod[t.item()].to(device=x_t.device, dtype=x_t.dtype)

            # Estimate clean image
            x_0_hat = x0_hat_from_eps(x_t, noise_pred, alpha_bar_t).clamp(-1, 1)

            # Classifier on estimated clean image
            logits = classifier(x_0_hat)
            probs = F.softmax(logits, dim=-1)
            target_prob = probs[0, target_digit]

            # Constraint: P(target) >= target_confidence
            h_tensor = target_confidence - target_prob

            # Dual update
            lmbda = torch.clamp(lmbda + lr_lambda * h_tensor.detach(), min=0.0)

            # Primal correction
            penalty = lmbda * h_tensor
            grad_x = torch.autograd.grad(penalty, x_t)[0]

            if grad_clip is not None:
                grad_x = torch.clamp(grad_x, -grad_clip, grad_clip)

            lambda_history.append(float(lmbda.item()))
            prob_history.append(float(target_prob.item()))

            with torch.no_grad():
                x_prev = scheduler.step(noise_pred.detach(), t, x_t.detach()).prev_sample
                x_t = x_prev - lr_x * grad_x.detach()

        else:
            with torch.no_grad():
                noise_pred = unet(x_t, t).sample
                x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    img_numpy = tensor_to_image(x_t)

    return GenerationResult(
        img_tensor=x_t.detach(),
        img_numpy=img_numpy,
        lambda_history=lambda_history,
        prob_history=prob_history,
        target_digit=target_digit,
        seed=seed,
        guided=use_pdlmc,
    )


def compare_guided_vs_baseline(
    unet,
    scheduler,
    classifier,
    device,
    target_digit=4,
    lr_lambda=0.01,
    lr_x=2.0,
    seed=78,
    num_steps=1000,
    target_confidence=0.9,
    grad_clip=0.05,
):
    """
    Generates baseline and guided outputs from the EXACT SAME seed/noise.
    """
    baseline = generate_digit(
        unet=unet,
        scheduler=scheduler,
        classifier=classifier,
        device=device,
        target_digit=target_digit,
        use_pdlmc=False,
        lr_lambda=lr_lambda,
        lr_x=lr_x,
        seed=seed,
        num_steps=num_steps,
        target_confidence=target_confidence,
        grad_clip=grad_clip,
    )

    guided = generate_digit(
        unet=unet,
        scheduler=scheduler,
        classifier=classifier,
        device=device,
        target_digit=target_digit,
        use_pdlmc=True,
        lr_lambda=lr_lambda,
        lr_x=lr_x,
        seed=seed,
        num_steps=num_steps,
        target_confidence=target_confidence,
        grad_clip=grad_clip,
    )

    return baseline, guided


# ==========================================================
# 5) Plots
# ==========================================================
def plot_generation_results(baseline, guided):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    axs[0].imshow(baseline.img_numpy, cmap="gray")
    axs[0].set_title(f"Baseline\nseed={baseline.seed}")
    axs[0].axis("off")

    axs[1].imshow(guided.img_numpy, cmap="gray")
    axs[1].set_title(f"Guided -> '{guided.target_digit}'\nseed={guided.seed}")
    axs[1].axis("off")

    axs[2].plot(guided.prob_history, color="green")
    axs[2].axhline(y=0.9, color="red", linestyle="--", label="Target (90%)")
    axs[2].set_title(f"Probability of class {guided.target_digit}")
    axs[2].set_xlabel("Steps")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    axs[3].plot(guided.lambda_history, color="purple")
    axs[3].set_title("Dual variable ($\lambda$)")
    axs[3].set_xlabel("Steps")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

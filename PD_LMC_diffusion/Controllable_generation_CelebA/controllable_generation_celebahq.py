import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from diffusers import DDPMPipeline
from tqdm.auto import tqdm


# ==========================================================
# Utils
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


def tensor_to_rgb_image(x):
    return (x / 2 + 0.5).clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


def make_initial_noise(unet, seed, device):
    """
    Creates initial Gaussian noise with the right DDPM shape.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    sample_size = unet.config.sample_size
    in_channels = unet.config.in_channels
    return torch.randn((1, in_channels, sample_size, sample_size), generator=g, device=device)


def x0_hat_from_eps(x_t, noise_pred, alpha_bar_t):
    """
    DDPM reconstruction of x0 from epsilon prediction.
    """
    return (x_t - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)


# ==========================================================
# Load DDPM
# ==========================================================
def load_celebahq_ddpm(
    model_id="google/ddpm-celebahq-256",
    device="cpu",
):
    pipe = DDPMPipeline.from_pretrained(model_id).to(device)
    return pipe, pipe.unet, pipe.scheduler


# ==========================================================
# Baseline generation
# ==========================================================
@dataclass
class FaceGenerationResult:
    img_tensor: torch.Tensor
    img_numpy: np.ndarray
    seed: int
    num_steps: int
    guided: bool
    history: dict


def generate_baseline_face(
    unet,
    scheduler,
    device,
    seed=42,
    num_steps=500,
):
    x_t = make_initial_noise(unet=unet, seed=seed, device=device)
    scheduler.set_timesteps(num_steps)

    for t in tqdm(scheduler.timesteps, desc=f"Baseline generation (seed={seed})", leave=False):
        with torch.no_grad():
            noise_pred = unet(x_t, t).sample
            x_t = scheduler.step(noise_pred, t, x_t).prev_sample

    return FaceGenerationResult(
        img_tensor=x_t.detach(),
        img_numpy=tensor_to_rgb_image(x_t),
        seed=seed,
        num_steps=num_steps,
        guided=False,
        history={},
    )


# ==========================================================
# Guided generation (brightness + structure)
# ==========================================================
def generate_guided_face(
    unet,
    scheduler,
    device,
    reference_tensor,
    seed=42,
    num_steps=500,
    target_brightness=0.75,
    structure_eps=0.005,
    lr_x=10.0,
    lr_lambda_light=20.0,
    lr_lambda_struct=10.0,
    grad_clip=0.01,
):
    """
    Multi-constraint training-free primal-dual guidance:
    - brightness constraint
    - structure constraint against a reference baseline image
    """
    x_t = make_initial_noise(unet=unet, seed=seed, device=device)
    scheduler.set_timesteps(num_steps)

    lmbda_light = torch.tensor(0.0, device=device)
    lmbda_struct = torch.tensor(0.0, device=device)

    history = {
        "lmbda_light": [],
        "lmbda_struct": [],
        "brightness": [],
        "mse": [],
    }

    reference_tensor = reference_tensor.detach().to(device)

    for t in tqdm(scheduler.timesteps, desc=f"Guided generation (seed={seed})", leave=False):
        # We need gradients through x_t and through the UNet
        x_t = x_t.detach().requires_grad_(True)

        noise_pred = unet(x_t, t).sample

        alpha_bar_t = scheduler.alphas_cumprod[t.item()].to(device=x_t.device, dtype=x_t.dtype)

        # Estimated clean image
        x_0_hat = x0_hat_from_eps(x_t, noise_pred, alpha_bar_t).clamp(-1, 1)

        # Brightness 
        x_0_hat_norm = (x_0_hat + 1) / 2
        current_brightness = x_0_hat_norm.mean()

        # Constraint 1: brightness >= target_brightness
        h_light = target_brightness - current_brightness

        # Constraint 2: MSE to baseline <= structure_eps
        mse_to_baseline = F.mse_loss(x_0_hat, reference_tensor)
        h_struct = mse_to_baseline - structure_eps

        # Dual updates
        lmbda_light = torch.clamp(lmbda_light + lr_lambda_light * h_light.detach(), min=0.0)
        lmbda_struct = torch.clamp(lmbda_struct + lr_lambda_struct * h_struct.detach(), min=0.0)

        # Primal correction
        penalty_total = lmbda_light * h_light + lmbda_struct * h_struct
        grad_x = torch.autograd.grad(penalty_total, x_t)[0]

        if grad_clip is not None:
            grad_x = torch.clamp(grad_x, -grad_clip, grad_clip)

        with torch.no_grad():
            x_prev = scheduler.step(noise_pred.detach(), t, x_t.detach()).prev_sample
            x_t = x_prev - lr_x * grad_x.detach()

        history["lmbda_light"].append(float(lmbda_light.item()))
        history["lmbda_struct"].append(float(lmbda_struct.item()))
        history["brightness"].append(float(current_brightness.item()))
        history["mse"].append(float(mse_to_baseline.item()))

    return FaceGenerationResult(
        img_tensor=x_t.detach(),
        img_numpy=tensor_to_rgb_image(x_t),
        seed=seed,
        num_steps=num_steps,
        guided=True,
        history=history,
    )


# ==========================================================
# Compare baseline vs guided from the SAME seed
# ==========================================================
def compare_baseline_vs_guided_faces(
    unet,
    scheduler,
    device,
    seed=50,
    num_steps=500,
    target_brightness=0.75,
    structure_eps=0.005,
    lr_x=10.0,
    lr_lambda_light=20.0,
    lr_lambda_struct=10.0,
    grad_clip=0.01,
):
    """
    Clean comparison:
    - baseline and guided start from the exact same initial noise
    - guided uses the baseline final image as structure reference
    """
    baseline = generate_baseline_face(
        unet=unet,
        scheduler=scheduler,
        device=device,
        seed=seed,
        num_steps=num_steps,
    )

    guided = generate_guided_face(
        unet=unet,
        scheduler=scheduler,
        device=device,
        reference_tensor=baseline.img_tensor,
        seed=seed,
        num_steps=num_steps,
        target_brightness=target_brightness,
        structure_eps=structure_eps,
        lr_x=lr_x,
        lr_lambda_light=lr_lambda_light,
        lr_lambda_struct=lr_lambda_struct,
        grad_clip=grad_clip,
    )

    return baseline, guided


# ==========================================================
# Plots
# ==========================================================
def plot_face_guidance_results(
    baseline,
    guided,
    target_brightness=0.75,
    structure_eps=0.005,
):
    hist = guided.history

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # Panel 1: baseline
    axs[0].imshow(baseline.img_numpy)
    axs[0].set_title(f"Baseline\nmean={baseline.img_numpy.mean():.3f}")
    axs[0].axis("off")

    # Panel 2: guided
    axs[1].imshow(guided.img_numpy)
    axs[1].set_title(f"Guided multi-objective\nmean={guided.img_numpy.mean():.3f}")
    axs[1].axis("off")

    # Panel 3: constraints tracking
    ax_left = axs[2]
    ax_left.plot(hist["brightness"], color="orange", label="Brightness")
    ax_left.axhline(y=target_brightness, color="red", linestyle="--", label="Brightness target")
    ax_left.set_title("Constraint fidelity")
    ax_left.set_xlabel("Steps")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(loc="upper left")

    ax_right = ax_left.twinx()
    ax_right.plot(hist["mse"], color="blue", alpha=0.6, label="MSE vs baseline")
    ax_right.axhline(y=structure_eps, color="navy", linestyle=":", label="Structure target")
    ax_right.legend(loc="upper right")

    # Panel 4: lambdas
    axs[3].plot(hist["lmbda_light"], color="red", label="lambda_light")
    axs[3].plot(hist["lmbda_struct"], color="purple", label="lambda_struct")
    axs[3].set_title("Dual variables")
    axs[3].set_xlabel("Steps")
    axs[3].grid(True, alpha=0.3)
    axs[3].legend()

    plt.tight_layout()
    plt.show()
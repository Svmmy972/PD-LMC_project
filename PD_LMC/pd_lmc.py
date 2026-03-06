import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .density import Density
from .constraint import Constraint


class PDLMC:
    """PD-LMC temporal chain sampling + plots."""

    def __init__(
        self,
        density: Density,
        constraint: Constraint,
        init_pos=None,
        dim=2,
        lr_x=1e-3,
        lr_lambda=0.5,
        num_particles=1, 
        device=None,
    ):
        self.density = density
        self.constraint = constraint
        self.dim = dim
        self.lr_x = lr_x
        self.lr_lambda = lr_lambda
        self.num_particles = num_particles
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Start point
        if init_pos is None:
            self.init_pos = torch.zeros(self.dim, device=self.device)
        else:
            self.init_pos = torch.tensor(init_pos, device=self.device, dtype=torch.float32)

        # Outputs
        self.chain = None            
        self.lambda_trace = None     
        self.samples = None          

    # ----------------------------
    # Algorithm
    # ----------------------------
    def run(
        self,
        num_steps=10000,
        slack=1e-3,
        lmc_steps_per_dual=1,
        store_every=1,
        seed=0,
        jitter=1e-2,
    ):
        """
        PD-LMC:
        """
        torch.manual_seed(seed)

        # init x:
        x = self.init_pos.detach().to(self.device).view(1, self.dim).repeat(self.num_particles, 1).clone()
        if jitter > 0:
            x = x + jitter * torch.randn_like(x)

        # global lambda scalar
        lam = torch.zeros((), device=self.device)

        self.lambda_trace = np.zeros((num_steps,), dtype=np.float32)

        saved = []
        sqrt_2eta = math.sqrt(2.0 * self.lr_x)

        for step in range(num_steps):
            g_acc = 0.0

            # inner steps (mini-batch in time)
            for _ in range(lmc_steps_per_dual):
                x_ = x.detach().requires_grad_(True)

                u = self.density.energy(x_)        
                h = self.constraint.evaluate(x_)   
                g = torch.relu(h) - slack          

                U = (u + lam * g).sum()
                grad_x = torch.autograd.grad(U, x_)[0]

                with torch.no_grad():
                    noise = torch.randn_like(x)
                    x = x - self.lr_x * grad_x + sqrt_2eta * noise

                    # measure g after move for dual update
                    h_next = self.constraint.evaluate(x)
                    g_next = torch.relu(h_next) - slack
                    g_acc += float(g_next.mean())

            with torch.no_grad():
                g_mean = g_acc / float(lmc_steps_per_dual)
                lam = torch.clamp(lam + self.lr_lambda * torch.tensor(g_mean, device=self.device), min=0.0)

            self.lambda_trace[step] = float(lam.detach().cpu())

            if (step % store_every) == 0:
                saved.append(x.detach().cpu().numpy())

        self.chain = np.stack(saved, axis=0)  # (T_saved,N, dim)
        return self.chain

    def collect_samples(self, burn_in=0, thin=1):
        """Post burn-in/thinning temporal samples."""
        self.samples = self.chain[burn_in:][::thin]
        return self.samples




    # ----------------------------
    # Rejection 
    # ----------------------------
    def rejection(self, n_draws=2_000_000, keep_for_plot=5000):
        """
        Simple rejection sampling 
        Draw n_draws from unconstrained density and keep those inside the constraint.
        """
        x = self.density.sample(n_draws, self.device)     
        h = self.constraint.evaluate(x)                   
        x_ok = x[h <= 0].detach().cpu().numpy()

        n_acc = x_ok.shape[0]
        acc_rate = n_acc / float(n_draws)

        if n_acc == 0:
            mean_ok = np.array([np.nan, np.nan])
            x_plot = x_ok
        else:
            mean_ok = x_ok.mean(axis=0)
            x_plot = x_ok[:keep_for_plot]

        return x_plot, mean_ok, acc_rate, n_acc

    # ----------------------------
    # Plot
    # ----------------------------
    def plot_results_elipsoid(
        self,
        filename="pdlmc_results.png",
        burn_in=0,
        thin=1,
        scatter_stride=10,
        limits=(-5, 5),
        truth_draws=200_000,   # rejection draws used to estimate "true mean"
        rej_draws=5000,          # points displayed in rejection panel
    ):
        """
        4 panels:
        1) Baseline unconstrained
        2) PD-LMC temporal samples
        3) Rejection samples (accepted)
        4) lmabda trace
        """
        if self.chain is None:
            raise RuntimeError("Run the sampler first.")

        # PD-LMC samples in time
        samples = self.collect_samples(burn_in=burn_in, thin=thin)
        sample_mean = samples.mean(axis=(0,1))

        # baseline cloud
        baseline = self.density.sample(5000, self.device).cpu().numpy()

        # rejection used for both panel + "true mean"
        rej_samples, true_mean, acc_rate, n_acc = self.rejection(
            n_draws=truth_draws,
            keep_for_plot=rej_draws
        )

        # ellipse params
        cx, cy = self.constraint.center.cpu().numpy()
        rx, ry = float(self.constraint.rx), float(self.constraint.ry)
        target_mean = self.density.mean.cpu().numpy()

        fig, axs = plt.subplots(1, 4, figsize=(24, 5))

        # Panel 1: Baseline
        ax0 = axs[0]
        ax0.scatter(baseline[:, 0], baseline[:, 1], alpha=0.25, s=6, color="tab:gray", label="Unconstrained")
        ax0.add_patch(Ellipse((cx, cy), rx * 2, ry * 2, edgecolor="r", fc="None", lw=2, linestyle="--", label="Constraint"))
        ax0.scatter(*target_mean, color="k", marker="x", s=120, label="Unconstrained mean")
        ax0.set(xlim=limits, ylim=limits, title="Baseline (no constraint)")
        ax0.grid(True, alpha=0.3)
        ax0.legend()

        # Panel 2: PD-LMC
        ax1 = axs[1]
        vis = samples[::scatter_stride, 0, :] if len(samples) > 0 else samples
        ax1.scatter(vis[:, 0], vis[:, 1], alpha=0.25, s=6, color="tab:blue", label="PD-LMC (temporal)")
        ax1.add_patch(Ellipse((cx, cy), rx * 2, ry * 2, edgecolor="r", fc="None", lw=2))
        ax1.scatter(*target_mean, color="k", marker="x", s=120, label="Unconstrained mean")
        ax1.scatter(*true_mean, color="red", marker="+", s=160, lw=3, label="True mean (rejection)")
        ax1.scatter(*sample_mean, color="orange", marker="+", s=160, lw=3, label="Sample mean")
        ax1.set(xlim=limits, ylim=limits, title="PD-LMC")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Panel 3: Rejection
        ax2 = axs[2]
        if rej_samples.shape[0] > 0:
            ax2.scatter(rej_samples[:, 0], rej_samples[:, 1], alpha=0.25, s=6, color="tab:green",
                        label=f"Accepted (n={rej_samples.shape[0]}/{n_acc})")
            ax2.scatter(*true_mean, color="orange", marker="+", s=160, lw=3, label="Rejection mean")
        else:
            ax2.text(0.5, 0.5, "No accepted samples", ha="center", va="center", transform=ax2.transAxes)

        ax2.add_patch(Ellipse((cx, cy), rx * 2, ry * 2, edgecolor="r", fc="None", lw=2))
        ax2.scatter(*target_mean, color="k", marker="x", s=120, label="Unconstrained mean")
        ax2.scatter(*true_mean, color="red", marker="+", s=160, lw=3, label="True mean (rejection)")
        ax2.set(xlim=limits, ylim=limits, title=f"Rejection (acc≈{acc_rate:.2e})")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Panel 4: lambda
        ax3 = axs[3]
        ax3.plot(self.lambda_trace, color="purple", linewidth=1.5)
        ax3.set(title="Dual variable λ", xlabel="Iteration", ylabel="λ")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=200)

        print(f"[Rejection] accepted_total={n_acc} / draws={truth_draws} => acc_rate≈{acc_rate:.3e}")
        print(f"True mean (rejection): {true_mean}")
        print(f"PD-LMC mean:          {sample_mean}")
        print(f"Saved to '{filename}'.")
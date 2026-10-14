import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import HACA3Dataset
from .network import ThetaEncoder3d
from .utils import mkdir_p, KLDivergenceLoss
import matplotlib.pyplot as plt


class ThetaModel:
    """
    Standalone pretraining model for HACA3+ theta.

    Goal
    ----
    Theta should represent MRI contrast rather than subject anatomy.

    Positive:
        same contrast
        different subjects

    Negative:
        different contrasts
    """

    def __init__(
        self,
        theta_dim=2,
        gpu_id=0,
        temperature=0.1,
        lambda_kld=1e-4,
    ):

        self.theta_dim = theta_dim
        self.temperature = temperature
        self.lambda_kld = lambda_kld

        self.device = torch.device(
            f"cuda:{gpu_id}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.timestr = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

        self.theta_encoder = ThetaEncoder3d(
            in_ch=1,
            out_ch=theta_dim,
        ).to(self.device)

        self.kld_loss = KLDivergenceLoss()

        self.optimizer = None

        self.train_loader = None
        self.valid_loader = None

        self.writer = None


    # ======================================================
    # DATA
    # ======================================================

    def load_dataset(
        self,
        dataset_dirs,
        contrasts,
        batch_size=4,
        normalization_method="01",
        num_workers=0,
        patch_size=None
    ):

        train_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="train",
            normalization_method=normalization_method,
            patch_size=patch_size,
        )

        valid_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="valid",
            normalization_method=normalization_method,
            patch_size=patch_size,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        self.contrasts = list(contrasts)

        print()
        print("===== THETA DATASET =====")
        print(f"Training samples:   {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")
        print()


    # ======================================================
    # INITIALIZE
    # ======================================================

    def initialize_training(
        self,
        out_dir,
        lr=1e-4,
    ):

        self.out_dir = Path(out_dir)

        mkdir_p(self.out_dir)

        self.model_dir = (
            self.out_dir
            / f"training_models_{self.timestr}"
        )
        self.result_dir = (
            self.out_dir
            / f"training_results_{self.timestr}"
        )
        
        mkdir_p(self.result_dir)
        mkdir_p(self.model_dir)

        self.writer = SummaryWriter(
            str(self.out_dir / self.timestr)
        )

        self.optimizer = Adam(
            self.theta_encoder.parameters(),
            lr=lr,
        )


    # ======================================================
    # COLLECT AVAILABLE IMAGES
    # ======================================================

    def collect_images(
        self,
        image_dicts,
    ):
        """
        Flatten available subject/contrast combinations into
        one image batch.

        Returns
        -------
        images:
            [M,1,D,H,W]

        contrast_ids:
            [M]
        """

        images = []
        contrast_ids = []

        for contrast_id, d in enumerate(
            image_dicts
        ):

            image = d["image"]
            exists = d["exists"]

            B = image.shape[0]

            for b in range(B):

                if exists[b].item() == 0:
                    continue

                images.append(
                    image[b:b+1]
                )

                contrast_ids.append(
                    contrast_id
                )

        if len(images) == 0:

            return None, None

        images = torch.cat(
            images,
            dim=0,
        ).to(self.device)

        contrast_ids = torch.tensor(
            contrast_ids,
            dtype=torch.long,
            device=self.device,
        )

        return (
            images,
            contrast_ids,
        )


    # ======================================================
    # SUPERVISED CONTRASTIVE LOSS
    # ======================================================

    def contrastive_loss(
        self,
        mu,
        contrast_ids,
    ):

        features = F.normalize(
            mu,
            dim=1,
        )

        similarity = torch.matmul(
            features,
            features.T,
        )

        similarity = (
            similarity
            / self.temperature
        )

        M = features.shape[0]

        self_mask = torch.eye(
            M,
            dtype=torch.bool,
            device=self.device,
        )

        positive_mask = (
            contrast_ids[:, None]
            ==
            contrast_ids[None, :]
        )

        positive_mask = (
            positive_mask
            & ~self_mask
        )

        # Remove self from denominator.

        logits = similarity.masked_fill(
            self_mask,
            float("-inf"),
        )

        log_prob = (
            logits
            - torch.logsumexp(
                logits,
                dim=1,
                keepdim=True,
            )
        )

        num_positive = (
            positive_mask.sum(
                dim=1
            )
        )

        valid = (
            num_positive > 0
        )

        positive_log_prob = (
            log_prob.masked_fill(
                ~positive_mask,
                0.0,
            ).sum(
                dim=1
            )
            /
            num_positive.clamp(
                min=1
            )
        )

        if valid.any():

            loss = (
                -positive_log_prob[
                    valid
                ].mean()
            )

        else:

            loss = (
                mu.sum() * 0.0
            )

        return loss


    # ======================================================
    # ONE EPOCH
    # ======================================================

    def run_epoch(
        self,
        loader,
        epoch,
        is_train=True,
        patch_plot_every=25,
    ):

        if is_train:
            self.theta_encoder.train()
        else:
            self.theta_encoder.eval()

        total_loss = 0.0
        total_contrastive = 0.0
        total_kld = 0.0

        num_batches = 0

        context = (
            torch.enable_grad()
            if is_train
            else torch.no_grad()
        )

        with context:

            progress = tqdm(
                loader,
                desc=(
                    f"Theta Train {epoch}"
                    if is_train
                    else f"Theta Valid {epoch}"
                ),
            )

            for batch_idx, image_dicts in enumerate(progress):

                (
                    images,
                    contrast_ids,
                ) = self.collect_images(
                    image_dicts
                )

                if images is None:
                    continue

                if is_train:

                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                # --------------------------------------
                # Encode
                # --------------------------------------

                mu, logvar = (
                    self.theta_encoder(
                        images
                    )
                )
                # --------------------------------------
                # Save patch/theta visualizations
                # --------------------------------------
                
                if (
                    patch_plot_every is not None
                    and patch_plot_every > 0
                    and (
                        batch_idx == 0
                        or (batch_idx + 1) % patch_plot_every == 0
                    )
                ):
                    self.save_patch_theta_visualization(
                        images=images,
                        mu=mu,
                        contrast_ids=contrast_ids,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        is_train=is_train,
                        max_images=8,
                    )

                # --------------------------------------
                # Contrastive
                # --------------------------------------

                contrastive = (
                    self.contrastive_loss(
                        mu,
                        contrast_ids,
                    )
                )

                # --------------------------------------
                # KLD
                # --------------------------------------

                kld = self.kld_loss(
                    mu,
                    logvar,
                ).mean()

                # --------------------------------------
                # Total
                # --------------------------------------

                loss = (
                    contrastive
                    +
                    self.lambda_kld
                    * kld
                )

                if is_train:

                    loss.backward()

                    self.optimizer.step()

                total_loss += loss.item()

                total_contrastive += (
                    contrastive.item()
                )

                total_kld += (
                    kld.item()
                )

                num_batches += 1

                progress.set_postfix(
                    total=f"{loss.item():.4f}",
                    con=f"{contrastive.item():.4f}",
                    kld=f"{kld.item():.4f}",
                )

        n = max(
            num_batches,
            1,
        )

        return {
            "loss":
                total_loss / n,

            "contrastive":
                total_contrastive / n,

            "kld":
                total_kld / n,
        }


    # ======================================================
    # TRAIN
    # ======================================================

    def train(
        self,
        num_epochs,
        save_every=100,
        plot_every=1,
        patch_plot_every=25,
    ):

        for epoch in range(
            1,
            num_epochs + 1,
        ):

            train_metrics = self.run_epoch(
                self.train_loader,
                epoch,
                is_train=True,
                patch_plot_every=patch_plot_every,
            )

            valid_metrics = self.run_epoch(
                self.valid_loader,
                epoch,
                is_train=False,
                patch_plot_every=patch_plot_every,
            )

            for name, value in train_metrics.items():

                self.writer.add_scalar(
                    f"train/{name}",
                    value,
                    epoch,
                )

            for name, value in valid_metrics.items():

                self.writer.add_scalar(
                    f"valid/{name}",
                    value,
                    epoch,
                )

            self.writer.add_scalar(
                "lr",
                self.optimizer.param_groups[0]["lr"],
                epoch,
            )

            print(
                f"Epoch {epoch} | "
                f"train={train_metrics['loss']:.4f} | "
                f"valid={valid_metrics['loss']:.4f} | "
                f"contrastive="
                f"{valid_metrics['contrastive']:.4f} | "
                f"KLD={valid_metrics['kld']:.4f}"
            )

            if (
                epoch % save_every == 0
                or epoch == num_epochs
            ):

                self.save_checkpoint(
                    epoch
                )
            if (
                epoch == 1
                or epoch % plot_every == 0
                or epoch == num_epochs
            ):
            
                self.save_theta_space(
                    epoch
                )


    # ======================================================
    # SAVE
    # ======================================================

    def save_checkpoint(
        self,
        epoch,
    ):

        checkpoint = {
            "epoch": epoch,

            "theta_encoder":
                self.theta_encoder.state_dict(),

            "optimizer":
                self.optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.model_dir
            / f"theta_model_{epoch}.pt",
        )
    def save_theta_space(
        self,
        epoch,
    ):
        """
        Save the 2D theta mu space for every available
        training and validation image.
    
        Color = MRI contrast
        Marker = train vs validation
        """
    
        if self.theta_dim != 2:
            print(
                "Skipping theta-space plot because "
                f"theta_dim={self.theta_dim}, not 2."
            )
            return
    
        self.theta_encoder.eval()
    
        all_mu = []
        all_contrast_ids = []
        all_splits = []
    
        loaders = [
            ("train", self.train_loader),
            ("valid", self.valid_loader),
        ]
    
        with torch.no_grad():
    
            for split_name, loader in loaders:
    
                for image_dicts in loader:
    
                    (
                        images,
                        contrast_ids,
                    ) = self.collect_images(
                        image_dicts
                    )
    
                    if images is None:
                        continue
    
                    mu, logvar = self.theta_encoder(
                        images
                    )
    
                    all_mu.append(
                        mu.cpu()
                    )
    
                    all_contrast_ids.append(
                        contrast_ids.cpu()
                    )
    
                    all_splits.extend(
                        [split_name] * mu.shape[0]
                    )
    
        # --------------------------------------------------
        # Combine everything
        # --------------------------------------------------
    
        all_mu = torch.cat(
            all_mu,
            dim=0,
        ).numpy()
    
        all_contrast_ids = torch.cat(
            all_contrast_ids,
            dim=0,
        ).numpy()
    
        # --------------------------------------------------
        # Plot
        # --------------------------------------------------
    
        fig, ax = plt.subplots(
            figsize=(9, 8)
        )
    
        colors = plt.cm.tab10.colors
    
        markers = {
            "train": "o",
            "valid": "^",
        }
    
        for contrast_id, contrast_name in enumerate(
            self.contrasts
        ):
    
            for split_name in [
                "train",
                "valid",
            ]:
    
                mask = (
                    (all_contrast_ids == contrast_id)
                    &
                    (
                        torch.tensor(
                            [
                                s == split_name
                                for s in all_splits
                            ]
                        ).numpy()
                    )
                )
    
                if mask.sum() == 0:
                    continue
    
                ax.scatter(
                    all_mu[mask, 0],
                    all_mu[mask, 1],
                    color=colors[
                        contrast_id
                        % len(colors)
                    ],
                    marker=markers[split_name],
                    alpha=0.65,
                    s=35,
                    label=(
                        f"{contrast_name} "
                        f"({split_name})"
                    ),
                )
    
        ax.set_xlabel(
            r"$\theta_1$"
        )
    
        ax.set_ylabel(
            r"$\theta_2$"
        )
    
        ax.set_title(
            f"Theta Space - Epoch {epoch}"
        )
    
        ax.axhline(
            0,
            linewidth=0.5,
            alpha=0.3,
        )
    
        ax.axvline(
            0,
            linewidth=0.5,
            alpha=0.3,
        )
    
        ax.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
    
        ax.grid(
            alpha=0.2
        )
    
        fig.tight_layout()
    
        # --------------------------------------------------
        # Save
        # --------------------------------------------------
    
        plot_path = (
            self.result_dir
            / f"theta_space_epoch_{epoch:05d}.png"
        )
    
        fig.savefig(
            plot_path,
            dpi=200,
            bbox_inches="tight",
        )
    
        plt.close(fig)
    
        print(
            f"Saved theta space: {plot_path}"
        )

    def save_patch_theta_visualization(
        self,
        images,
        mu,
        contrast_ids,
        epoch,
        batch_idx,
        is_train=True,
        max_images=8,
    ):
        """
        Save center axial slices of 3D patches with their
        corresponding theta values.
    
        images:       [M, 1, D, H, W]
        mu:           [M, theta_dim]
        contrast_ids: [M]
        """
    
        split = "train" if is_train else "valid"
    
        save_dir = self.result_dir / "patch_theta" / split
        mkdir_p(save_dir)
    
        images = images.detach().cpu()
        mu = mu.detach().cpu()
        contrast_ids = contrast_ids.detach().cpu()
    
        n_images = min(max_images, images.shape[0])
    
        if n_images == 0:
            return
    
        ncols = min(4, n_images)
        nrows = (n_images + ncols - 1) // ncols
    
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4 * ncols, 4 * nrows),
            squeeze=False,
        )
    
        for i in range(nrows * ncols):
    
            ax = axes.flat[i]
    
            if i >= n_images:
                ax.axis("off")
                continue
    
            # [D, H, W]
            volume = images[i, 0]
    
            # Center axial slice
            slice_idx = volume.shape[2] // 2
    
            image_slice = volume[:, :, slice_idx].numpy()
    
            contrast_id = int(contrast_ids[i])
            contrast_name = self.contrasts[contrast_id]
    
            theta_values = mu[i].numpy()
    
            ax.imshow(
                image_slice.T,
                cmap="gray",
                origin="lower",
            )
    
            ax.set_title(
                f"{contrast_name}\n"
                f"theta1 = {theta_values[0]:.4f}\n"
                f"theta2 = {theta_values[1]:.4f}",
                fontsize=11,
            )
    
            ax.axis("off")
    
        fig.suptitle(
            f"Theta Patches | {split} | "
            f"Epoch {epoch} | Batch {batch_idx}",
            fontsize=14,
        )
    
        fig.tight_layout()
    
        save_path = (
            save_dir
            / f"theta_patches_epoch_{epoch:05d}"
              f"_batch_{batch_idx:05d}.png"
        )
    
        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )
    
        plt.close(fig)

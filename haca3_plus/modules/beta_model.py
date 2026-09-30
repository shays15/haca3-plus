import os
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import HACA3Dataset
from .network import UNet3d
from .utils import mkdir_p, reparameterize_logit, save_image_3d


class BetaModel:
    """
    Standalone pretraining model for the HACA3+ beta encoder.

    Goal
    ----
    Learn a spatial anatomical representation that is invariant
    to MRI contrast.

    Positive pair:
        same subject/session
        different MRI contrast
        same spatial location

    Negative examples:
        other spatial locations within the same subject.
    """

    def __init__(
        self,
        beta_dim=5,
        gpu_id=0,
        temperature=0.1,
        patch_size=16,
    ):

        self.beta_dim = beta_dim
        self.temperature = temperature
        self.patch_size = patch_size
        self.fixed_valid_pair = None

        self.device = torch.device(
            f"cuda:{gpu_id}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.timestr = datetime.now().strftime(
            "%Y%m%d-%H%M%S"
        )

        # --------------------------------------------------
        # Same beta encoder architecture as HACA3+
        # --------------------------------------------------

        self.beta_encoder = UNet3d(
            in_ch=1,
            out_ch=beta_dim,
            base_ch=8,
            final_act="none",
        ).to(self.device)

        self.optimizer = None

        self.train_loader = None
        self.valid_loader = None

        self.writer = None
        self.out_dir = None


    # ======================================================
    # DATA
    # ======================================================

    def load_dataset(
        self,
        dataset_dirs,
        contrasts,
        batch_size=1,
        normalization_method="01",
        num_workers=0,
    ):

        train_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="train",
            normalization_method=normalization_method,
        )

        valid_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="valid",
            normalization_method=normalization_method,
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

        print()
        print("===== BETA DATASET =====")
        print(f"Training samples:   {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")
        print()


    # ======================================================
    # INITIALIZE TRAINING
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

        mkdir_p(self.model_dir)
        mkdir_p(self.result_dir)

        self.writer = SummaryWriter(
            str(self.out_dir / self.timestr)
        )

        self.optimizer = Adam(
            self.beta_encoder.parameters(),
            lr=lr,
        )

    def get_fixed_validation_pair(self):
    
        if self.fixed_valid_pair is not None:
            return self.fixed_valid_pair
    
        for image_dicts in self.valid_loader:
    
            available = torch.stack(
                [
                    d["exists"]
                    for d in image_dicts
                ],
                dim=1,
            )
    
            B = available.shape[0]
    
            for b in range(B):
    
                ids = torch.where(
                    available[b] > 0
                )[0].tolist()
    
                if len(ids) < 2:
                    continue
    
                # --------------------------------------------------
                # Prefer T1PRE + another contrast
                # --------------------------------------------------
    
                if 0 in ids:
    
                    contrast_a = 0
    
                    other_ids = [
                        i
                        for i in ids
                        if i != 0
                    ]
    
                    contrast_b = other_ids[0]
    
                else:
    
                    contrast_a = ids[0]
                    contrast_b = ids[1]
    
                image_a = (
                    image_dicts[
                        contrast_a
                    ]["image"][b:b+1]
                    .to(self.device)
                )
    
                image_b = (
                    image_dicts[
                        contrast_b
                    ]["image"][b:b+1]
                    .to(self.device)
                )
    
                self.fixed_valid_pair = (
                    image_a,
                    image_b,
                    contrast_a,
                    contrast_b,
                )
    
                print(
                    "Fixed beta validation pair:",
                    contrast_a,
                    contrast_b,
                )
    
                return self.fixed_valid_pair
    
        raise RuntimeError(
            "Could not find a validation subject "
            "with at least two available contrasts."
        )
    # ======================================================
    # BETA REPRESENTATION
    # ======================================================

    def calculate_beta(
        self,
        image,
    ):
        """
        image:
            [B,1,D,H,W]

        Returns
        -------
        logits:
            [B,beta_dim,D,H,W]

        probabilities:
            softmax representation used for contrastive training

        beta:
            final HACA3+ scalar beta map
            [B,1,D,H,W]
        """

        logits = self.beta_encoder(
            image
        )

        # ----------------------------------------------
        # Continuous representation for training
        # ----------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=1,
        )

        # ----------------------------------------------
        # Original HACA3+ hard representation
        # ----------------------------------------------

        beta_onehot = reparameterize_logit(
            logits
        )

        beta = self.channel_aggregation(
            beta_onehot
        )

        return (
            logits,
            probabilities,
            beta,
        )


    def channel_aggregation(
        self,
        beta_onehot,
    ):

        value_tensor = torch.arange(
            self.beta_dim,
            device=beta_onehot.device,
            dtype=beta_onehot.dtype,
        )

        value_tensor = value_tensor.view(
            1,
            self.beta_dim,
            1,
            1,
            1,
        )

        beta = (
            beta_onehot
            * value_tensor
        ).sum(
            dim=1,
            keepdim=True,
        )

        beta = beta / self.beta_dim

        return beta


    # ======================================================
    # PATCH FEATURES
    # ======================================================

    def get_patch_features(
        self,
        probabilities,
    ):
        """
        probabilities:
            [B,C,D,H,W]

        Average the beta probability distribution within
        non-overlapping spatial patches.

        Output:
            [B,N,C]

        No trainable projection network is used.
        """

        p = self.patch_size

        features = F.avg_pool3d(
            probabilities,
            kernel_size=p,
            stride=p,
        )

        # [B,C,d,h,w]
        # ->
        # [B,N,C]

        features = features.flatten(
            start_dim=2
        ).transpose(
            1,
            2
        )

        features = F.normalize(
            features,
            dim=-1,
        )

        return features


    # ======================================================
    # CONTRASTIVE LOSS
    # ======================================================

    def contrastive_loss(
        self,
        probabilities_a,
        probabilities_b,
    ):

        feature_a = self.get_patch_features(
            probabilities_a
        )

        feature_b = self.get_patch_features(
            probabilities_b
        )

        # ----------------------------------------------
        # Cross-contrast spatial similarity
        # ----------------------------------------------

        logits_ab = torch.bmm(
            feature_a,
            feature_b.transpose(1, 2),
        )

        logits_ab = (
            logits_ab
            / self.temperature
        )

        B, N, _ = logits_ab.shape

        targets = torch.arange(
            N,
            device=self.device,
        )

        targets = targets.unsqueeze(0).expand(
            B,
            -1,
        )

        # A -> B

        loss_ab = F.cross_entropy(
            logits_ab.reshape(
                B * N,
                N,
            ),
            targets.reshape(
                B * N
            ),
        )

        # B -> A

        logits_ba = logits_ab.transpose(
            1,
            2
        )

        loss_ba = F.cross_entropy(
            logits_ba.reshape(
                B * N,
                N,
            ),
            targets.reshape(
                B * N
            ),
        )

        return 0.5 * (
            loss_ab + loss_ba
        )


    # ======================================================
    # DIRECT BETA METRICS
    # ======================================================

    @staticmethod
    def beta_metrics(
        beta_a,
        beta_b,
    ):
    
        # ======================================================
        # CROSS-CONTRAST MAE
        # ======================================================
    
        mae = F.l1_loss(
            beta_a,
            beta_b,
        )
    
    
        # ======================================================
        # CROSS-CONTRAST CORRELATION
        # ======================================================
    
        a = beta_a.flatten(
            start_dim=1
        )
    
        b = beta_b.flatten(
            start_dim=1
        )
    
        a_centered = (
            a
            - a.mean(
                dim=1,
                keepdim=True,
            )
        )
    
        b_centered = (
            b
            - b.mean(
                dim=1,
                keepdim=True,
            )
        )
    
        correlation = (
            (a_centered * b_centered).sum(dim=1)
            /
            (
                torch.sqrt(
                    (a_centered ** 2).sum(dim=1)
                )
                *
                torch.sqrt(
                    (b_centered ** 2).sum(dim=1)
                )
                + 1e-8
            )
        ).mean()
    
    
        # ======================================================
        # SPATIAL VARIANCE
        # ======================================================
    
        std_a = a.std(
            dim=1
        ).mean()
    
        std_b = b.std(
            dim=1
        ).mean()
    
        beta_std = 0.5 * (
            std_a + std_b
        )
    
    
        return (
            mae,
            correlation,
            beta_std,
        )


    # ======================================================
    # SELECT CONTRAST PAIR
    # ======================================================

    def select_contrast_pair(
        self,
        image_dicts,
    ):

        available = torch.stack(
            [
                d["exists"]
                for d in image_dicts
            ],
            dim=1,
        )

        pairs = []

        B = available.shape[0]

        for b in range(B):

            ids = torch.where(
                available[b] > 0
            )[0].tolist()

            if len(ids) < 2:
                continue

            a, c = random.sample(
                ids,
                2,
            )

            image_a = (
                image_dicts[a]["image"][b:b+1]
                .to(self.device)
            )

            image_b = (
                image_dicts[c]["image"][b:b+1]
                .to(self.device)
            )

            pairs.append(
                (
                    image_a,
                    image_b,
                    a,
                    c,
                )
            )

        return pairs


    # ======================================================
    # ONE EPOCH
    # ======================================================

    def run_epoch(
        self,
        loader,
        epoch,
        is_train=True,
    ):

        if is_train:
            self.beta_encoder.train()
        else:
            self.beta_encoder.eval()

        total_loss = 0.0
        num_batches = 0
        total_mae = 0.0
        total_corr = 0.0
        num_pairs = 0
        total_std = 0.0

        context = (
            torch.enable_grad()
            if is_train
            else torch.no_grad()
        )

        with context:

            progress = tqdm(
                loader,
                desc=(
                    f"Beta Train {epoch}"
                    if is_train
                    else f"Beta Valid {epoch}"
                ),
            )

            for image_dicts in progress:

                pairs = self.select_contrast_pair(
                    image_dicts
                )

                if len(pairs) == 0:
                    continue

                if is_train:
                    self.optimizer.zero_grad(
                        set_to_none=True
                    )

                losses = []

                for (
                    image_a,
                    image_b,
                    contrast_a,
                    contrast_b,
                ) in pairs:

                    (
                        logits_a,
                        probabilities_a,
                        beta_a,
                    ) = self.calculate_beta(
                        image_a
                    )

                    (
                        logits_b,
                        probabilities_b,
                        beta_b,
                    ) = self.calculate_beta(
                        image_b
                    )

                    loss = self.contrastive_loss(
                        probabilities_a,
                        probabilities_b,
                    )

                    mae, corr, beta_std = self.beta_metrics(
                        beta_a,
                        beta_b,
                    )
                    
                    total_mae += mae.item()
                    total_corr += corr.item()
                    total_std += beta_std.item()
                    num_pairs += 1

                    losses.append(loss)

                loss = torch.stack(
                    losses
                ).mean()

                if is_train:

                    loss.backward()

                    self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                progress.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        n_batches = max(
            len(loader),
            1,
        )

        mean_loss = total_loss / max(num_batches, 1)

        mean_mae = (
            total_mae
            / max(num_pairs, 1)
        )

        mean_corr = (
            total_corr
            / max(num_pairs, 1)
        )

        return {
        "loss": mean_loss,
        "beta_mae": (
            total_mae / max(num_pairs, 1)
        ),
        "beta_corr": (
            total_corr / max(num_pairs, 1)
        ),
        "beta_std": (
            total_std / max(num_pairs, 1)
        ),
    }


    # ======================================================
    # TRAIN
    # ======================================================

    def train(
        self,
        num_epochs,
        save_every=100,
        image_every=10,
    ):        

        for epoch in range(
            1,
            num_epochs + 1,
        ):

            train_metrics = self.run_epoch(
                self.train_loader,
                epoch,
                is_train=True,
            )

            valid_metrics = self.run_epoch(
                self.valid_loader,
                epoch,
                is_train=False,
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

            print(
                f"Epoch {epoch} | "
                f"train={train_metrics['loss']:.4f} | "
                f"valid={valid_metrics['loss']:.4f} | "
                f"MAE={valid_metrics['beta_mae']:.4f} | "
                f"corr={valid_metrics['beta_corr']:.4f}"
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
                or epoch % image_every == 0
                or epoch == num_epochs
            ):
            
                self.save_validation_images(
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
            "beta_encoder":
                self.beta_encoder.state_dict(),
            "optimizer":
                self.optimizer.state_dict(),
        }

        torch.save(
            checkpoint,
            self.model_dir
            / f"beta_model_{epoch}.pt",
        )

    def save_validation_images(
        self,
        epoch,
    ):
    
        (
            image_a,
            image_b,
            contrast_a,
            contrast_b,
        ) = self.get_fixed_validation_pair()
    
    
        self.beta_encoder.eval()
    
        with torch.no_grad():
    
            (
                logits_a,
                probabilities_a,
                beta_a,
            ) = self.calculate_beta(
                image_a
            )
    
            (
                logits_b,
                probabilities_b,
                beta_b,
            ) = self.calculate_beta(
                image_b
            )
    
    
            # ==================================================
            # DIFFERENCE MAP
            # ==================================================
    
            beta_difference = torch.abs(
                beta_a - beta_b
            )
    
    
            # ==================================================
            # SAVE NIFTI
            # ==================================================
    
            output_path = (
                self.result_dir
                / (
                    f"beta_epoch_{epoch:05d}"
                    f"_c{contrast_a}_c{contrast_b}.nii.gz"
                )
            )
    
            save_image_3d(
                [
                    image_a,
                    image_b,
                    beta_a,
                    beta_b,
                    beta_difference,
                ],
                str(output_path),
            )

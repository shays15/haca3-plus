import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import HACA3Dataset
from .network import UNet3d
from .utils import mkdir_p, save_image_3d


class BetaModel:
    """
    Standalone pretraining model for the HACA3+ beta encoder.

    Goal
    ----
    Learn a beta representation that:
        1. preserves anatomy well enough to reconstruct T1PRE, and
        2. is similar across MRI contrasts from the same subject/session.

    Training pair
    -------------
    For each subject/session:
        T1PRE -> beta -> decoder -> T1PRE
        other contrast -> beta -> decoder -> T1PRE

    Loss
    ----
        total_loss = reconstruction_loss
                   + lambda_beta * beta_consistency_loss

    The reconstruction term prevents beta from satisfying the
    cross-contrast objective with a non-anatomical or collapsed code.

    The beta-consistency term is computed on the deterministic softmax
    probability representation, not on stochastic Gumbel samples.
    """

    def __init__(
        self,
        beta_dim=5,
        gpu_id=0,
        lambda_beta=0.1,
        temperature=None,
        patch_size=None,
    ):
        self.beta_dim = beta_dim
        self.lambda_beta = lambda_beta

        # Kept only so the existing train_beta.py interface does not break.
        # They are not used by this reconstruction-based beta objective.
        self.temperature = temperature
        self.patch_size = patch_size

        self.device = torch.device(
            f"cuda:{gpu_id}"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Same beta encoder architecture as HACA3+.
        self.beta_encoder = UNet3d(
            in_ch=1,
            out_ch=beta_dim,
            base_ch=8,
            final_act="none",
        ).to(self.device)

        # Pretraining-only decoder.
        #
        # It consumes the beta probability volume directly. This forces
        # the representation that receives the consistency loss to retain
        # enough spatial/anatomical information to reconstruct T1PRE.
        self.decoder = UNet3d(
            in_ch=beta_dim,
            out_ch=1,
            base_ch=8,
            final_act="relu",
        ).to(self.device)

        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None

        self.writer = None
        self.out_dir = None
        self.model_dir = None
        self.result_dir = None

        self.contrasts = None
        self.t1_index = None
        self.fixed_valid_pair = None


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
        self.contrasts = list(contrasts)

        if "T1PRE" not in self.contrasts:
            raise ValueError(
                "Beta pretraining requires T1PRE because T1PRE is "
                "used as the anatomical reconstruction target."
            )

        self.t1_index = self.contrasts.index("T1PRE")

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
        print(f"T1PRE index:        {self.t1_index}")
        print(f"Lambda beta:        {self.lambda_beta}")
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

        # The decoder is trained jointly during beta pretraining,
        # but only beta_encoder needs to be transferred into HACA3+.
        self.optimizer = Adam(
            list(self.beta_encoder.parameters())
            + list(self.decoder.parameters()),
            lr=lr,
        )


    # ======================================================
    # BETA REPRESENTATION
    # ======================================================

    def calculate_beta(self, image):
        """
        Returns
        -------
        logits:
            [B, beta_dim, D, H, W]
    
        probabilities:
            [B, beta_dim, D, H, W]
    
        beta:
            [B, 1, D, H, W]
            Probability-weighted scalar beta representation.
        """

        logits = self.beta_encoder(image)
    
        probabilities = F.softmax(
            logits,
            dim=1,
        )
    
        # Channel values: [0, 1, 2, 3, 4]
        channel_values = torch.arange(
            self.beta_dim,
            device=probabilities.device,
            dtype=probabilities.dtype,
        ).view(
            1, self.beta_dim, 1, 1, 1
        )
    
        # Weighted aggregation across beta channels
        beta = (
            probabilities * channel_values
        ).sum(
            dim=1,
            keepdim=True,
        ) / self.beta_dim
    
        return logits, probabilities, beta

    def decode_beta(self, probabilities):
        """
        Reconstruct T1PRE directly from the beta probability volume.
        """
        return self.decoder(probabilities)

    # ======================================================
    # LOSSES
    # ======================================================

    @staticmethod
    def reconstruction_loss(reconstruction, target):
        return F.l1_loss(
            reconstruction,
            target,
        )


    @staticmethod
    def beta_consistency_loss(probabilities_a, probabilities_b):
        """
        Encourage the same subject/session to have the same beta
        probability representation across contrasts.
        """
        return F.l1_loss(
            probabilities_a,
            probabilities_b,
        )


    # ======================================================
    # METRICS
    # ======================================================

    @staticmethod
    def beta_metrics(
        probabilities_a,
        probabilities_b,
        beta_a,
        beta_b,
    ):
        # --------------------------------------------------
        # Soft probability MAE
        # --------------------------------------------------
        probability_mae = F.l1_loss(
            probabilities_a,
            probabilities_b,
        )

        # --------------------------------------------------
        # Hard/scalar beta MAE
        # --------------------------------------------------
        beta_mae = F.l1_loss(
            beta_a,
            beta_b,
        )

        # --------------------------------------------------
        # Scalar beta correlation
        # --------------------------------------------------
        a = beta_a.flatten(start_dim=1)
        b = beta_b.flatten(start_dim=1)

        a_centered = a - a.mean(
            dim=1,
            keepdim=True,
        )

        b_centered = b - b.mean(
            dim=1,
            keepdim=True,
        )

        numerator = (
            a_centered
            * b_centered
        ).sum(dim=1)

        denominator = (
            torch.sqrt(
                (a_centered ** 2).sum(dim=1)
            )
            *
            torch.sqrt(
                (b_centered ** 2).sum(dim=1)
            )
        )

        correlation = (
            numerator
            / (denominator + 1e-8)
        ).mean()

        # --------------------------------------------------
        # Spatial beta std
        # --------------------------------------------------
        beta_std = 0.5 * (
            a.std(dim=1).mean()
            + b.std(dim=1).mean()
        )

        # --------------------------------------------------
        # Probability entropy
        #
        # Very low entropy everywhere can indicate hard channel
        # assignments; very high entropy near log(beta_dim) means
        # near-uniform probabilities. This is a diagnostic only.
        # --------------------------------------------------
        eps = 1e-8

        entropy_a = -(
            probabilities_a
            * torch.log(
                probabilities_a + eps
            )
        ).sum(dim=1).mean()

        entropy_b = -(
            probabilities_b
            * torch.log(
                probabilities_b + eps
            )
        ).sum(dim=1).mean()

        probability_entropy = 0.5 * (
            entropy_a + entropy_b
        )

        return {
            "probability_mae": probability_mae,
            "beta_mae": beta_mae,
            "beta_corr": correlation,
            "beta_std": beta_std,
            "probability_entropy": probability_entropy,
        }


    # ======================================================
    # SELECT T1 + OTHER-CONTRAST PAIRS
    # ======================================================

    def select_training_pairs(self, image_dicts):
        """
        For each subject in the batch, select:
            target/input A = T1PRE
            input B        = one random available non-T1 contrast

        Subjects without T1PRE or without another contrast are skipped.

        One non-T1 contrast is sampled per subject per iteration so that
        memory usage stays close to two forward passes per subject rather
        than loading every contrast through the encoder simultaneously.
        """
        if self.t1_index is None:
            raise RuntimeError(
                "load_dataset() must be called before training."
            )

        pairs = []

        batch_size = image_dicts[self.t1_index]["image"].shape[0]

        for b in range(batch_size):
            t1_exists = (
                image_dicts[self.t1_index]["exists"][b].item()
                > 0
            )

            if not t1_exists:
                continue

            available_other_ids = []

            for contrast_id, image_dict in enumerate(image_dicts):
                if contrast_id == self.t1_index:
                    continue

                if image_dict["exists"][b].item() > 0:
                    available_other_ids.append(
                        contrast_id
                    )

            if len(available_other_ids) == 0:
                continue

            other_id = random.choice(
                available_other_ids
            )

            t1_image = (
                image_dicts[self.t1_index]["image"][b:b + 1]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            other_image = (
                image_dicts[other_id]["image"][b:b + 1]
                .to(
                    self.device,
                    non_blocking=True,
                )
            )

            pairs.append(
                (
                    t1_image,
                    other_image,
                    self.t1_index,
                    other_id,
                )
            )

        return pairs


    # ======================================================
    # FIXED VALIDATION PAIR
    # ======================================================

    def get_fixed_validation_pair(self):
        """
        Always return the same validation subject and contrast pair
        for saved visualizations.
        """
        if self.fixed_valid_pair is not None:
            return self.fixed_valid_pair

        if self.valid_loader is None:
            raise RuntimeError(
                "Validation loader has not been initialized."
            )

        for image_dicts in self.valid_loader:
            batch_size = (
                image_dicts[self.t1_index]["image"].shape[0]
            )

            for b in range(batch_size):
                if (
                    image_dicts[self.t1_index]["exists"][b].item()
                    == 0
                ):
                    continue

                other_ids = []

                for contrast_id, image_dict in enumerate(
                    image_dicts
                ):
                    if contrast_id == self.t1_index:
                        continue

                    if image_dict["exists"][b].item() > 0:
                        other_ids.append(
                            contrast_id
                        )

                if len(other_ids) == 0:
                    continue

                other_id = other_ids[0]

                t1_image = (
                    image_dicts[self.t1_index]["image"][b:b + 1]
                    .to(self.device)
                )

                other_image = (
                    image_dicts[other_id]["image"][b:b + 1]
                    .to(self.device)
                )

                self.fixed_valid_pair = (
                    t1_image,
                    other_image,
                    self.t1_index,
                    other_id,
                )

                print(
                    "Fixed beta validation pair:",
                    self.contrasts[self.t1_index],
                    "vs",
                    self.contrasts[other_id],
                )

                return self.fixed_valid_pair

        raise RuntimeError(
            "Could not find a validation subject with T1PRE "
            "and at least one additional contrast."
        )


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
            self.decoder.train()
        else:
            self.beta_encoder.eval()
            self.decoder.eval()

        totals = {
            "loss": 0.0,
            "recon_loss": 0.0,
            "recon_t1": 0.0,
            "recon_other": 0.0,
            "beta_loss": 0.0,
            "probability_mae": 0.0,
            "beta_mae": 0.0,
            "beta_corr": 0.0,
            "beta_std": 0.0,
            "probability_entropy": 0.0,
        }

        num_pairs = 0

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
                pairs = self.select_training_pairs(
                    image_dicts
                )

                if len(pairs) == 0:
                    continue

                # With the expected beta batch size of 1 this loop has
                # one pair. Keeping the loop makes the code work for
                # larger subject batches too.
                for (
                    t1_image,
                    other_image,
                    t1_id,
                    other_id,
                ) in pairs:
                    if is_train:
                        self.optimizer.zero_grad(
                            set_to_none=True
                        )

                    # ------------------------------------------
                    # Encode T1PRE
                    # ------------------------------------------
                    (
                        logits_t1,
                        probabilities_t1,
                        beta_t1,
                    ) = self.calculate_beta(
                        t1_image
                    )

                    # ------------------------------------------
                    # Encode another contrast
                    # ------------------------------------------
                    (
                        logits_other,
                        probabilities_other,
                        beta_other,
                    ) = self.calculate_beta(
                        other_image
                    )

                    # ------------------------------------------
                    # Reconstruct the SAME T1PRE target from
                    # both beta representations.
                    # ------------------------------------------
                    reconstruction_t1 = self.decode_beta(
                        probabilities_t1
                    )

                    reconstruction_other = self.decode_beta(
                        probabilities_other
                    )

                    loss_recon_t1 = (
                        self.reconstruction_loss(
                            reconstruction_t1,
                            t1_image,
                        )
                    )

                    loss_recon_other = (
                        self.reconstruction_loss(
                            reconstruction_other,
                            t1_image,
                        )
                    )

                    loss_recon = 0.5 * (
                        loss_recon_t1
                        + loss_recon_other
                    )

                    # ------------------------------------------
                    # Cross-contrast beta consistency
                    # ------------------------------------------
                    loss_beta = (
                        self.beta_consistency_loss(
                            probabilities_t1,
                            probabilities_other,
                        )
                    )

                    # ------------------------------------------
                    # Total
                    # ------------------------------------------
                    loss = (
                        loss_recon
                        + self.lambda_beta
                        * loss_beta
                    )

                    if is_train:
                        loss.backward()
                        self.optimizer.step()

                    # ------------------------------------------
                    # Diagnostics
                    # ------------------------------------------
                    metrics = self.beta_metrics(
                        probabilities_t1.detach(),
                        probabilities_other.detach(),
                        beta_t1.detach(),
                        beta_other.detach(),
                    )

                    totals["loss"] += loss.item()
                    totals["recon_loss"] += (
                        loss_recon.item()
                    )
                    totals["recon_t1"] += (
                        loss_recon_t1.item()
                    )
                    totals["recon_other"] += (
                        loss_recon_other.item()
                    )
                    totals["beta_loss"] += (
                        loss_beta.item()
                    )

                    for name, value in metrics.items():
                        totals[name] += value.item()

                    num_pairs += 1

                    progress.set_postfix(
                        total=f"{loss.item():.4f}",
                        recon=f"{loss_recon.item():.4f}",
                        beta=f"{loss_beta.item():.4f}",
                    )

        denominator = max(
            num_pairs,
            1,
        )

        return {
            name: value / denominator
            for name, value in totals.items()
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

            # ----------------------------------------------
            # TensorBoard
            # ----------------------------------------------
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
                f"recon={valid_metrics['recon_loss']:.4f} | "
                f"beta={valid_metrics['beta_loss']:.4f} | "
                f"beta MAE={valid_metrics['beta_mae']:.4f} | "
                f"corr={valid_metrics['beta_corr']:.4f} | "
                f"std={valid_metrics['beta_std']:.4f}"
            )

            # ----------------------------------------------
            # Save visual validation output
            # ----------------------------------------------
            if (
                epoch == 1
                or epoch % image_every == 0
                or epoch == num_epochs
            ):
                self.save_validation_images(
                    epoch
                )

            # ----------------------------------------------
            # Save checkpoint
            # ----------------------------------------------
            if (
                epoch % save_every == 0
                or epoch == num_epochs
            ):
                self.save_checkpoint(
                    epoch
                )

        if self.writer is not None:
            self.writer.flush()


    # ======================================================
    # SAVE CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        epoch,
    ):
        checkpoint = {
            "epoch": epoch,
            "beta_encoder":
                self.beta_encoder.state_dict(),
            "decoder":
                self.decoder.state_dict(),
            "optimizer":
                self.optimizer.state_dict(),
            "beta_dim":
                self.beta_dim,
            "lambda_beta":
                self.lambda_beta,
            "contrasts":
                self.contrasts,
        }

        torch.save(
            checkpoint,
            self.model_dir
            / f"beta_model_{epoch}.pt",
        )


    # ======================================================
    # SAVE VALIDATION IMAGES
    # ======================================================

    def save_validation_images(
        self,
        epoch,
    ):
        (
            t1_image,
            other_image,
            t1_id,
            other_id,
        ) = self.get_fixed_validation_pair()

        beta_was_training = self.beta_encoder.training
        decoder_was_training = self.decoder.training

        self.beta_encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            (
                logits_t1,
                probabilities_t1,
                beta_t1,
            ) = self.calculate_beta(
                t1_image
            )

            (
                logits_other,
                probabilities_other,
                beta_other,
            ) = self.calculate_beta(
                other_image
            )

            reconstruction_t1 = self.decode_beta(
                probabilities_t1
            )

            reconstruction_other = self.decode_beta(
                probabilities_other
            )

            beta_difference = torch.abs(
                beta_t1
                - beta_other
            )

            probability_difference = torch.mean(
                torch.abs(
                    probabilities_t1
                    - probabilities_other
                ),
                dim=1,
                keepdim=True,
            )

            # ----------------------------------------------
            # Overview NIfTI
            #
            # volume 0: T1PRE input / target
            # volume 1: other-contrast input
            # volume 2: reconstruction from T1 beta
            # volume 3: reconstruction from other beta
            # volume 4: scalar beta from T1
            # volume 5: scalar beta from other contrast
            # volume 6: |scalar beta T1 - scalar beta other|
            # volume 7: mean channel-wise probability difference
            # ----------------------------------------------
            overview_path = (
                self.result_dir
                / (
                    f"beta_epoch_{epoch:05d}"
                    f"_{self.contrasts[t1_id]}"
                    f"_vs_{self.contrasts[other_id]}"
                    f"_overview.nii.gz"
                )
            )

            save_image_3d(
                [
                    t1_image,
                    other_image,
                    reconstruction_t1,
                    reconstruction_other,
                    beta_t1,
                    beta_other,
                    beta_difference,
                    probability_difference,
                ],
                str(overview_path),
            )

            # ----------------------------------------------
            # Save all soft beta probability channels.
            #
            # These are the representation actually used by
            # the reconstruction and consistency objectives.
            # ----------------------------------------------
            t1_probability_volumes = [
                probabilities_t1[:, i:i + 1]
                for i in range(self.beta_dim)
            ]

            other_probability_volumes = [
                probabilities_other[:, i:i + 1]
                for i in range(self.beta_dim)
            ]

            probability_path = (
                self.result_dir
                / (
                    f"beta_epoch_{epoch:05d}"
                    f"_{self.contrasts[t1_id]}"
                    f"_vs_{self.contrasts[other_id]}"
                    f"_probabilities.nii.gz"
                )
            )

            save_image_3d(
                t1_probability_volumes
                + other_probability_volumes,
                str(probability_path),
            )

        if beta_was_training:
            self.beta_encoder.train()

        if decoder_was_training:
            self.decoder.train()

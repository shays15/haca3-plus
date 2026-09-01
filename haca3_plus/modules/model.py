from tqdm import tqdm
import numpy as np
import random
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.transforms import ToTensor
from datetime import datetime
import nibabel as nib
from torch.cuda.amp import autocast
from pathlib import Path

from .utils import *
from .dataset import HACA3Dataset
from .network import (
    UNet3d,
    ThetaEncoder3d,
    EtaEncoder3d,
    Patchifier3d,
    AttentionModule3d,
)

class HACA3:
    def __init__(self, beta_dim, theta_dim, eta_dim, pretrained_haca3=None, pretrained_eta_encoder=None, gpu_id=0):
        self.beta_dim = beta_dim
        self.theta_dim = theta_dim
        self.eta_dim = eta_dim
        self.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        self.timestr = datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_loader, self.valid_loader = None, None
        self.out_dir = None
        self.optimizer = None
        self.scheduler = None
        self.writer, self.writer_path = None, None
        self.checkpoint = None

        self.l1_loss, self.kld_loss, self.contrastive_loss, self.perceptual_loss = None, None, None, None

        # define networks
        self.beta_encoder = UNet3d(
            in_ch=1,
            out_ch=self.beta_dim,
            base_ch=8,
            final_act='none'
        )
        
        self.theta_encoder = ThetaEncoder3d(
            in_ch=1,
            out_ch=self.theta_dim
        )
        
        self.eta_encoder = EtaEncoder3d(
            in_ch=1,
            out_ch=self.eta_dim
        )
        
        self.attention_module = AttentionModule3d(
            self.theta_dim + self.eta_dim,
            v_ch=self.beta_dim
        )
        
        self.decoder = UNet3d(
            in_ch=1 + self.theta_dim,
            out_ch=1,
            base_ch=8,   # start with 8
            final_act='relu'
        )
        
        self.patchifier = Patchifier3d(
            in_ch=1,
            out_ch=128
        )

        if pretrained_eta_encoder is not None:
            checkpoint_eta_encoder = torch.load(pretrained_eta_encoder, map_location=self.device)
            self.eta_encoder.load_state_dict(checkpoint_eta_encoder['eta_encoder'])
        if pretrained_haca3 is not None:
            self.checkpoint = torch.load(pretrained_haca3, map_location=self.device)
            self.beta_encoder.load_state_dict(self.checkpoint['beta_encoder'])
            self.theta_encoder.load_state_dict(self.checkpoint['theta_encoder'])
            self.eta_encoder.load_state_dict(self.checkpoint['eta_encoder'])
            self.decoder.load_state_dict(self.checkpoint['decoder'])
            self.attention_module.load_state_dict(self.checkpoint['attention_module'])
            self.patchifier.load_state_dict(self.checkpoint['patchifier'])
        self.beta_encoder.to(self.device)
        self.theta_encoder.to(self.device)
        self.eta_encoder.to(self.device)
        self.decoder.to(self.device)
        self.attention_module.to(self.device)
        self.patchifier.to(self.device)
        self.start_epoch = 0

    def initialize_training(self, out_dir, lr):
        # define loss functions
        self.l1_loss = nn.L1Loss(reduction='none')
        self.kld_loss = KLDivergenceLoss()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.to(self.device)
        self.perceptual_loss = PerceptualLoss(vgg)
        self.contrastive_loss = PatchNCELoss()

        # define optimizer and learning rate scheduler
        self.optimizer = Adam(list(self.beta_encoder.parameters()) +
                              list(self.theta_encoder.parameters()) +
                              list(self.decoder.parameters()) +
                              list(self.attention_module.parameters()) +
                              list(self.patchifier.parameters()), lr=lr)
        self.scheduler = CyclicLR(self.optimizer, base_lr=4e-4, max_lr=7e-4, cycle_momentum=False)
        if self.checkpoint is not None:
            self.start_epoch = self.checkpoint['epoch']
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])
            if 'timestr' in self.checkpoint:
                self.timestr = self.checkpoint['timestr']
        self.start_epoch = self.start_epoch + 1
        self.scaler = torch.cuda.amp.GradScaler()

        self.out_dir = out_dir
        mkdir_p(self.out_dir)
        mkdir_p(os.path.join(self.out_dir, f'training_results_{self.timestr}'))
        mkdir_p(os.path.join(self.out_dir, f'training_models_{self.timestr}'))

        self.writer_path = os.path.join(self.out_dir, self.timestr)
        self.writer = SummaryWriter(self.writer_path)

    def load_dataset(
        self,
        dataset_dirs,
        contrasts,
        batch_size=1,
        normalization_method="01",
        num_workers=0,
    ):
        """
        Load full-volume 3D HACA3+ training and validation datasets.
    
        Each contrast returned by the DataLoader has shape:
            [B, 1, D, H, W]
    
        where currently:
            D, H, W = 192, 224, 192
        """
    
        # ======================================================
        # TRAINING DATASET
        # ======================================================
    
        train_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="train",
            normalization_method=normalization_method,
        )
    
    
        # ======================================================
        # VALIDATION DATASET
        # ======================================================
    
        valid_dataset = HACA3Dataset(
            dataset_dirs=dataset_dirs,
            contrasts=contrasts,
            mode="valid",
            normalization_method=normalization_method,
        )
    
    
        # ======================================================
        # PRINT DATASET SIZES
        # ======================================================
    
        print()
        print("===== DATASET SIZES =====")
        print(f"Training samples:   {len(train_dataset)}")
        print(f"Validation samples: {len(valid_dataset)}")
        print()
    
    
        if len(train_dataset) == 0:
            raise RuntimeError(
                "Training dataset contains zero samples."
            )
    
        if len(valid_dataset) == 0:
            raise RuntimeError(
                "Validation dataset contains zero samples."
            )
    
    
        # ======================================================
        # DATA LOADERS
        # ======================================================
    
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

    def calculate_theta(self, images):
        if isinstance(images, list):
            thetas, mus, logvars = [], [], []
            for image in images:
                mu, logvar = self.theta_encoder(image)
                theta = torch.randn(mu.size()).to(self.device) * torch.sqrt(torch.exp(logvar)) + mu
                thetas.append(theta)
                mus.append(mu)
                logvars.append(logvar)
        else:
            mus, logvars = self.theta_encoder(images)
            thetas = torch.randn(mus.size()).to(self.device) * torch.sqrt(torch.exp(logvars)) + mus
        return thetas, mus, logvars

    def calculate_beta(self, images):
        logits, betas = [], []
        for image in images:
            logit = self.beta_encoder(image)
            beta = self.channel_aggregation(reparameterize_logit(logit))
            logits.append(logit)
            betas.append(beta)
        return logits, betas

    def calculate_eta(self, images):
        if isinstance(images, list):
            etas = []
            for image in images:
                eta = self.eta_encoder(image)
                etas.append(eta)
        else:
            etas = self.eta_encoder(images)
        return etas

    def prepare_source_images(self, image_dicts):
        num_contrasts = len(image_dicts)
        num_contrasts_with_degradation = np.random.permutation(num_contrasts)[0]
        degradation_ids = sorted(np.random.choice(range(num_contrasts),
                                                  num_contrasts_with_degradation,
                                                  replace=False))
        source_images = []
        for i in range(num_contrasts):
            if i in degradation_ids:
                source_images.append(image_dicts[i]['image_degrade'].to(self.device))
            else:
                source_images.append(image_dicts[i]['image'].to(self.device))
        return source_images

    def channel_aggregation(self, beta_onehot_encode):
        """
        beta_onehot_encode:
            [B, beta_dim, D, H, W]
    
        returns:
            [B, 1, D, H, W]
        """
    
        value_tensor = torch.arange(
            self.beta_dim,
            device=beta_onehot_encode.device,
            dtype=beta_onehot_encode.dtype
        )
    
        value_tensor = value_tensor.view(
            1,
            self.beta_dim,
            1,
            1,
            1
        )
    
        beta_label_encode = (
            beta_onehot_encode
            * value_tensor
        )
    
        return (
            beta_label_encode.sum(
                dim=1,
                keepdim=True
            )
            / self.beta_dim
        )

    def select_available_contrasts(self, image_dicts):
        """
        Select available contrasts as target.

        ===INPUTS===
        * image_dicts: list (num_contrasts, )
            List of dictionaries. Each element is a dictionary received from dataloader. See dataset.py for details.

        ===OUTPUTS===
        * target_image: torch.Tensor (batch_size, 1, image_dim=224, image_dim=224)
            Images as target for I2I.
        *  selected_contrast_id: torch.Tensor (batch_size, num_contrasts)
            Indicates which contrast has been selected as target image.
        """
        target_image_combined = torch.cat([d['image'] for d in image_dicts], dim=1)
        # (batch_size, num_contrasts)
        available_contrasts = torch.stack([d['exists'] for d in image_dicts], dim=-1)
        subject_ids = available_contrasts.nonzero(as_tuple=True)[0]
        contrast_ids = available_contrasts.nonzero(as_tuple=True)[1]
        unique_subject_ids = list(torch.unique(subject_ids))
        selected_contrast_ids = []
        for i in unique_subject_ids:
            selected_contrast_ids.append(random.choice(contrast_ids[subject_ids == i]))
        target_image = target_image_combined[unique_subject_ids, selected_contrast_ids, ...].unsqueeze(1).to(
            self.device)
        selected_contrast_id = torch.zeros_like(available_contrasts).to(self.device)
        selected_contrast_id[unique_subject_ids, selected_contrast_ids, ...] = 1.0
        return target_image, selected_contrast_id

    def decode(
        self,
        logits,
        target_theta,
        query,
        keys,
        available_contrast_id,
        mask=None,
        contrast_dropout=False,
        contrast_id_to_drop=None,
    ):
        """
        logits:
            list of N tensors:
            [B, beta_dim, D, H, W]
    
        target_theta:
            [B, theta_dim]
    
        query:
            [B, theta_dim + eta_dim]
    
        keys:
            list of N tensors:
            each [B, theta_dim + eta_dim]
    
        available_contrast_id:
            [B, N]
    
        mask:
            [B, N, D, H, W]
        """
    
        # --------------------------------
        # Stack beta logits
        # --------------------------------
    
        # [B, N, beta_dim, D, H, W]
        v = torch.stack(
            logits,
            dim=1
        )
    
        # --------------------------------
        # Stack source keys
        # --------------------------------
    
        # list N × [B,4]
        #
        # ->
        #
        # [B,N,4]
    
        k = torch.stack(
            keys,
            dim=1
        )
    
        # q already:
        # [B,4]
        q = query
    
        # --------------------------------
        # Contrast dropout
        # --------------------------------
    
        if contrast_dropout:
    
            available_contrast_id = dropout_contrasts(
                available_contrast_id,
                contrast_id_to_drop
            )
    
        # --------------------------------
        # Attention
        # --------------------------------
    
        logit_fusion, attention = self.attention_module(
            q,
            k,
            v,
            mask,
            modality_dropout=1 - available_contrast_id,
            temperature=10.0
        )
    
        # logit_fusion:
        # [B, beta_dim, D, H, W]
    
        # attention:
        # [B, N, D, H, W]
    
        # --------------------------------
        # Beta
        # --------------------------------
    
        beta_fusion = self.channel_aggregation(
            reparameterize_logit(
                logit_fusion
            )
        )
    
        # [B,1,D,H,W]
    
        # --------------------------------
        # Broadcast target theta
        # --------------------------------
    
        B, _, D, H, W = beta_fusion.shape
    
        target_theta_map = target_theta[
            :,
            :,
            None,
            None,
            None
        ].expand(
            -1,
            -1,
            D,
            H,
            W
        )
    
        # [B,theta_dim,D,H,W]
    
        # --------------------------------
        # Decoder
        # --------------------------------
    
        combined_map = torch.cat(
            [
                beta_fusion,
                target_theta_map
            ],
            dim=1
        )
    
        # [B, 1 + theta_dim, D,H,W]
    
        rec_image = self.decoder(
            combined_map
        )
    
        return (
            rec_image,
            attention,
            logit_fusion,
            beta_fusion
        )

    def calculate_features_for_contrastive_loss(
        self,
        betas,
        source_images,
        available_contrast_id,
    ):
    
        # Each list contains N tensors of:
        # [B, 1, D, H, W]
        #
        # Stack modality dimension:
        # [B, N, 1, D, H, W]
    
        betas_stack = torch.stack(betas, dim=1)
        source_images_stack = torch.stack(source_images, dim=1)
    
        B, N = available_contrast_id.shape
    
        query_features = []
        positive_features = []
        negative_features = []
    
        for subject_id in range(B):
    
            available_ids = torch.where(
                available_contrast_id[subject_id] > 0
            )[0]
    
            # Pick one available modality as the query
            query_idx = available_ids[
                torch.randint(
                    len(available_ids),
                    (1,),
                    device=available_ids.device
                )
            ].item()
    
            # --------------------------------
            # Query
            # --------------------------------
    
            query_image = source_images_stack[
                subject_id:subject_id + 1,
                query_idx
            ]
            #print("source_images_stack:", source_images_stack.shape)
            #print("betas_stack:", betas_stack.shape)
    
            query_feature = self.patchifier(
                query_image
            ).flatten(start_dim=2)
    
            # --------------------------------
            # Positive
            # beta from same modality
            # --------------------------------
    
            positive_beta = betas_stack[
                subject_id:subject_id + 1,
                query_idx
            ]
            #print("query image:", query_image.shape)
            #print("positive beta:", positive_beta.shape)
            
            positive_feature = self.patchifier(
                positive_beta
            ).flatten(start_dim=2)
    
            # --------------------------------
            # Negatives
            # other available modalities
            # --------------------------------
    
            subject_negative_features = []
    
            for contrast_idx in available_ids:
    
                contrast_idx = contrast_idx.item()
    
                if contrast_idx == query_idx:
                    continue
    
                negative_beta = betas_stack[
                    subject_id:subject_id + 1,
                    contrast_idx
                ]
    
                negative_feature = self.patchifier(
                    negative_beta
                ).flatten(start_dim=2)
    
                subject_negative_features.append(
                    negative_feature
                )
    
            if len(subject_negative_features) > 0:
    
                negative_feature = torch.cat(
                    subject_negative_features,
                    dim=2
                )
    
            else:
                negative_feature = positive_feature
    
            query_features.append(query_feature)
            positive_features.append(positive_feature)
            negative_features.append(negative_feature)
    
        query_features = torch.cat(
            query_features,
            dim=0
        )
    
        positive_features = torch.cat(
            positive_features,
            dim=0
        )
    
        negative_features = torch.cat(
            negative_features,
            dim=0
        )
    
        return (
            query_features,
            positive_features,
            negative_features,
        )

    def calculate_loss(
        self,
        rec_image,
        ref_image,
        mask,
        mu,
        logvar,
        betas,
        source_images,
        available_contrast_id,
        is_train=True,
    ):
        """
        Calculate the main HACA3+ losses for full-volume 3D
        training and validation.
    
        Current losses:
            reconstruction L1
            KLD
            beta PatchNCE
    
        The original 2D VGG perceptual loss is disabled because
        VGG16 cannot directly operate on 3D volumes.
    
        Cycle consistency is handled separately and is currently
        disabled while testing batch-size-1 3D training.
        """
    
        # ======================================================
        # 1. RECONSTRUCTION LOSS
        # ======================================================
    
        # Ensure boolean mask
        mask = mask.bool()
    
        rec_loss = self.l1_loss(
            rec_image[mask],
            ref_image[mask],
        ).mean()
    
    
        # ======================================================
        # 2. PERCEPTUAL LOSS
        # ======================================================
        #
        # Original HACA3+ used a 2D VGG16 perceptual loss.
        # Do not apply that directly to:
        #
        #     [B, C, D, H, W]
        #
        # For now this term is disabled.
        # ======================================================
    
        perceptual_loss = torch.tensor(
            0.0,
            device=rec_image.device,
        )
    
    
        # ======================================================
        # 3. KLD LOSS
        # ======================================================
    
        kld_loss = self.kld_loss(
            mu,
            logvar,
        ).mean()
    
    
        # ======================================================
        # 4. BETA PATCHNCE LOSS
        # ======================================================
    
        (
            query_feature,
            positive_feature,
            negative_feature,
        ) = self.calculate_features_for_contrastive_loss(
            betas,
            source_images,
            available_contrast_id,
        )
    
    
        beta_loss = self.contrastive_loss(
            query_feature,
            positive_feature.detach(),
            negative_feature.detach(),
        )
    
    
        # ======================================================
        # 5. TOTAL LOSS
        # ======================================================
    
        total_loss = (
            10.0 * rec_loss
            + 1e-5 * kld_loss
            + 5e-1 * beta_loss
        )
    
    
        # ======================================================
        # 6. OPTIMIZATION
        # ======================================================
    
        if is_train:
            self.optimizer.zero_grad(
                set_to_none=True
            )
        
            self.scaler.scale(
                total_loss
            ).backward()
        
            self.scaler.step(
                self.optimizer
            )
        
            self.scaler.update()
        
            self.scheduler.step()
    
        # ======================================================
        # 7. RETURN LOSSES
        # ======================================================
    
        loss = {
            "rec_loss": rec_loss.item(),
            "per_loss": perceptual_loss.item(),
            "kld_loss": kld_loss.item(),
            "beta_loss": beta_loss.item(),
            "total_loss": total_loss.item(),
        }
    
        return loss

    def calculate_cycle_consistency_loss(self, theta_rec, theta_ref, eta_rec, eta_ref, beta_rec, beta_ref,
                                         is_train=True):
        theta_loss = self.l1_loss(theta_rec, theta_ref).mean()
        eta_loss = self.l1_loss(eta_rec, eta_ref).mean()
        beta_loss = self.l1_loss(beta_rec, beta_ref).mean()

        cycle_loss = theta_loss + eta_loss + 5e-2 * beta_loss
        if is_train:
            self.optimizer.zero_grad()
            (5e-2 * cycle_loss).backward()
            self.optimizer.step()
            self.scheduler.step()
        loss = {'theta_cyc': theta_loss.item(),
                'eta_cyc': eta_loss.item(),
                'beta_cyc': beta_loss.item()}
        return loss

    def write_tensorboard(self, loss, epoch, batch_id, train_or_valid='train', cycle_loss=None):
        if train_or_valid == 'train':
            curr_iteration = (epoch - 1) * len(self.train_loader) + batch_id
            self.writer.add_scalar(f'{train_or_valid}/learning rate', self.scheduler.get_last_lr()[0], curr_iteration)
        else:
            curr_iteration = (epoch - 1) * len(self.valid_loader) + batch_id
        self.writer.add_scalar(f'{train_or_valid}/reconstruction loss', loss['rec_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/perceptual loss', loss['per_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/kld loss', loss['kld_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/beta loss', loss['beta_loss'], curr_iteration)
        self.writer.add_scalar(f'{train_or_valid}/total loss', loss['total_loss'], curr_iteration)
        if cycle_loss is not None:
            self.writer.add_scalar(f'{train_or_valid}/theta cycle loss', cycle_loss['theta_cyc'], curr_iteration)
            self.writer.add_scalar(f'{train_or_valid}/eta cycle loss', cycle_loss['eta_cyc'], curr_iteration)
            self.writer.add_scalar(f'{train_or_valid}/beta cycle loss', cycle_loss['beta_cyc'], curr_iteration)

    def save_model(self, epoch, file_name):
        state = {'epoch': epoch,
                 'timestr': self.timestr,
                 'beta_encoder': self.beta_encoder.state_dict(),
                 'theta_encoder': self.theta_encoder.state_dict(),
                 'eta_encoder': self.eta_encoder.state_dict(),
                 'decoder': self.decoder.state_dict(),
                 'attention_module': self.attention_module.state_dict(),
                 'patchifier': self.patchifier.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'scheduler': self.scheduler.state_dict()}
        torch.save(obj=state, f=file_name)

    def image_to_image_translation(
        self,
        batch_id,
        epoch,
        image_dicts,
        train_or_valid,
    ):
        """
        One full-volume 3D HACA3+ intra-site training/validation step.
    
        Each image:
            [B, 1, D, H, W]
    
        masks:
            [B, N, D, H, W]
    
        available_contrast_id:
            [B, N]
        """
    
        # ======================================================
        # TRAIN / VALID SETTINGS
        # ======================================================
    
        is_train = (
            train_or_valid == "train"
        )
    
        contrast_dropout = (
            True if is_train else False
        )
    
    
        # ======================================================
        # PREPARE SOURCE IMAGES
        # ======================================================
    
        source_images = (
            self.prepare_source_images(
                image_dicts
            )
        )
    
        # Move source images to GPU explicitly in case
        # prepare_source_images does not already do this.
        source_images = [
            image.to(
                self.device,
                non_blocking=True,
            )
            for image in source_images
        ]
    
    
        # ======================================================
        # MASKS
        # ======================================================
    
        # Target/reconstruction mask:
        # [B,1,D,H,W]
        mask = (
            image_dicts[0]["mask"]
            .to(
                self.device,
                non_blocking=True,
            )
        )
    
    
        # Each DataLoader mask:
        # [B,1,D,H,W]
        #
        # Remove image-channel dimension before stacking:
        #
        # [B,D,H,W] x N
        #       ->
        # [B,N,D,H,W]
        masks = torch.stack(
            [
                d["mask"].squeeze(1)
                for d in image_dicts
            ],
            dim=1,
        ).to(
            self.device,
            non_blocking=True,
        )
    
    
        assert masks.ndim == 5, (
            f"Expected masks [B,N,D,H,W], "
            f"got {masks.shape}"
        )
    
        assert masks.shape[1] == len(
            image_dicts
        )
    
    
        # ======================================================
        # SELECT TARGET CONTRAST
        # ======================================================
    
        (
            target_image,
            contrast_id_for_decoding,
        ) = self.select_available_contrasts(
            image_dicts
        )
    
        target_image = target_image.to(
            self.device,
            non_blocking=True,
        )
    
    
        # ======================================================
        # AVAILABLE CONTRASTS
        #
        # Example:
        # T1 / T2 / PD / FLAIR
        #  1    1    0     1
        #
        # shape: [B,N]
        # ======================================================
    
        available_contrast_id = torch.stack(
            [
                d["exists"]
                for d in image_dicts
            ],
            dim=1,
        ).to(
            self.device,
            non_blocking=True,
        )
    
    
        assert available_contrast_id.ndim == 2
    
        assert available_contrast_id.shape[1] == len(
            image_dicts
        )
    
    
        # ======================================================
        # FORWARD PASS
        # ======================================================
    
        with torch.cuda.amp.autocast():
    
            # ==================================================
            # BETA
            # ==================================================
    
            logits, betas = (
                self.calculate_beta(
                    source_images
                )
            )
    
    
            # ==================================================
            # SOURCE THETA / ETA
            # ==================================================
    
            (
                thetas_source,
                _,
                _,
            ) = self.calculate_theta(
                source_images
            )
    
            etas_source = (
                self.calculate_eta(
                    source_images
                )
            )
    
    
            # ==================================================
            # TARGET THETA / ETA
            # ==================================================
    
            (
                theta_target,
                mu_target,
                logvar_target,
            ) = self.calculate_theta(
                target_image
            )
    
            eta_target = (
                self.calculate_eta(
                    target_image
                )
            )
    
    
            # ==================================================
            # ATTENTION QUERY / KEYS
            # ==================================================
    
            query = torch.cat(
                [
                    theta_target,
                    eta_target,
                ],
                dim=1,
            )
    
    
            keys = [
                torch.cat(
                    [
                        theta,
                        eta,
                    ],
                    dim=1,
                )
                for (
                    theta,
                    eta,
                ) in zip(
                    thetas_source,
                    etas_source,
                )
            ]
    
    
            # ==================================================
            # CONTRAST DROPOUT
            # ==================================================
    
            if (
                is_train
                and torch.rand(
                    1
                ).item() > 0.2
            ):
    
                contrast_id_to_drop = (
                    contrast_id_for_decoding
                )
    
            else:
    
                contrast_id_to_drop = None
    
    
            # ==================================================
            # DECODE FULL 3D VOLUME
            # ==================================================
    
            (
                rec_image,
                attention,
                logit_fusion,
                beta_fusion,
            ) = self.decode(
                logits,
                theta_target,
                query,
                keys,
                available_contrast_id,
                masks,
                contrast_dropout=contrast_dropout,
                contrast_id_to_drop=contrast_id_to_drop,
            )
    
    
            # ==================================================
            # LOSS
            # ==================================================
    
            loss = self.calculate_loss(
                rec_image,
                target_image,
                mask,
                mu_target,
                logvar_target,
                betas,
                source_images,
                available_contrast_id,
                is_train=is_train,
            )
    
    
        # ======================================================
        # SAVE TRAINING EXAMPLES
        # ======================================================
    
        if (
            batch_id != 10
        ):
    
            file_name = os.path.join(
                self.out_dir,
                f"training_results_{self.timestr}",
                (
                    f"{train_or_valid}_"
                    f"epoch{str(epoch).zfill(3)}_"
                    f"batch{str(batch_id).zfill(4)}_"
                    f"intra-site.nii.gz"
                ),
            )
    
            save_image_3d(
                source_images
                + [rec_image]
                + [target_image]
                + betas
                + [beta_fusion],
                file_name,
            )
    
    
        # ======================================================
        # INTER-SITE CYCLE TRAINING
        # ======================================================
        #
        # IMPORTANT:
        #
        # Old implementation used:
        #
        #     torch.randperm(batch_size)
        #
        # to shuffle target images within the batch.
        #
        # With 3D training batch_size = 1:
        #
        #     randperm(1) == [0]
        #
        # so the "shuffled" image would actually be the SAME
        # subject.
        #
        # Therefore inter-site cycle training is intentionally
        # disabled here until we add an independent target
        # sampler.
        # ======================================================
    
        cycle_loss = None
    
    
        # ======================================================
        # TENSORBOARD
        # ======================================================
    
        self.write_tensorboard(
            loss,
            epoch,
            batch_id,
            train_or_valid,
        )
    
    
        # ======================================================
        # SAVE MODEL
        # ======================================================
    
        if (
            batch_id % 2000 == 0
            and is_train
        ):
    
            file_name = os.path.join(
                self.out_dir,
                f"training_models_{self.timestr}",
                (
                    f"epoch{str(epoch).zfill(3)}_"
                    f"batch{str(batch_id).zfill(4)}_"
                    f"model.pt"
                ),
            )
    
            self.save_model(
                epoch,
                file_name,
            )
    
    
        # ======================================================
        # RETURN LOSSES FOR TQDM
        # ======================================================
    
        return loss

    def train(
        self,
        epochs,
    ):
    
        for epoch in range(
            self.start_epoch,
            epochs + 1,
        ):
    
            print()
            print(
                f"========== EPOCH {epoch}/{epochs} =========="
            )
    
    
            # ==================================================
            # TRAINING
            # ==================================================
    
            self.beta_encoder.train()
            self.theta_encoder.train()
    
            # Eta stays frozen
            self.eta_encoder.eval()
    
            self.decoder.train()
            self.attention_module.train()
            self.patchifier.train()
    
    
            train_iterator = tqdm(
                self.train_loader,
                desc=f"Train {epoch}/{epochs}",
            )
    
    
            for (
                batch_id,
                image_dicts,
            ) in enumerate(
                train_iterator
            ):
    
                loss = self.image_to_image_translation(
                    batch_id,
                    epoch,
                    image_dicts,
                    train_or_valid="train",
                )
    
    
                train_iterator.set_description(
                    (
                        f"Train {epoch}/{epochs} | "
                        f"rec {loss['rec_loss']:.3f} | "
                        f"kld {loss['kld_loss']:.3f} | "
                        f"beta {loss['beta_loss']:.3f}"
                    )
                )
    
    
            # ==================================================
            # VALIDATION
            # ==================================================
    
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()
            self.attention_module.eval()
            self.patchifier.eval()
    
    
            valid_iterator = tqdm(
                self.valid_loader,
                desc=f"Valid {epoch}/{epochs}",
            )
    
    
            with torch.no_grad():
    
                for (
                    batch_id,
                    image_dicts,
                ) in enumerate(
                    valid_iterator
                ):
    
                    loss = self.image_to_image_translation(
                        batch_id,
                        epoch,
                        image_dicts,
                        train_or_valid="valid",
                    )
    
    
                    valid_iterator.set_description(
                        (
                            f"Valid {epoch}/{epochs} | "
                            f"rec {loss['rec_loss']:.3f} | "
                            f"kld {loss['kld_loss']:.3f} | "
                            f"beta {loss['beta_loss']:.3f}"
                        )
                    )
    def harmonize(
        self,
        source_images,
        target_images,
        target_theta,
        target_eta,
        out_paths,
        affine,
        header,
        norm_vals,
        save_intermediate=False,
        intermediate_out_dir=None,
    ):
        """
        Perform full-volume 3D HACA3+ harmonization.
    
        Parameters
        ----------
        source_images:
            List of source volumes.
            Each tensor has shape:
                [1, 1, D, H, W]
    
        target_images:
            Optional list of target images.
            Each tensor:
                [1, 1, D, H, W]
    
        target_theta:
            Optional manually specified theta:
                [N_targets, theta_dim]
    
        target_eta:
            Optional manually specified eta:
                [N_targets, eta_dim]
    
        out_paths:
            Output NIfTI paths.
    
        affine:
            Reference NIfTI affine.
    
        header:
            Reference NIfTI header.
    
        norm_vals:
            Values used to return outputs to desired
            intensity scale.
        """
    
        # ======================================================
        # EVAL MODE
        # ======================================================
    
        self.beta_encoder.eval()
        self.theta_encoder.eval()
        self.eta_encoder.eval()
        self.attention_module.eval()
        self.decoder.eval()
    
        device = self.device
    
        # ======================================================
        # MOVE SOURCE IMAGES TO GPU
        # ======================================================
    
        source_images = [
            image.to(
                device,
                non_blocking=True,
            )
            for image in source_images
        ]
    
        print(
            "Source shapes:",
            [tuple(x.shape) for x in source_images],
        )
    
        # ======================================================
        # FORWARD SOURCE ENCODERS
        # ======================================================
    
        with torch.inference_mode():
    
            with torch.cuda.amp.autocast():
    
                # ------------------------------------------------
                # β anatomy representation
                # ------------------------------------------------
    
                logits, betas = self.calculate_beta(
                    source_images
                )
    
                # ------------------------------------------------
                # θ contrast representation
                # ------------------------------------------------
    
                thetas_source, _, _ = self.calculate_theta(
                    source_images
                )
    
                # ------------------------------------------------
                # η artifact representation
                # ------------------------------------------------
    
                etas_source = self.calculate_eta(
                    source_images
                )
    
                # ------------------------------------------------
                # Attention keys
                #
                # each key:
                # [B, theta_dim + eta_dim]
                # ------------------------------------------------
    
                keys = [
                    torch.cat(
                        [
                            theta,
                            eta,
                        ],
                        dim=1,
                    )
                    for theta, eta in zip(
                        thetas_source,
                        etas_source,
                    )
                ]
                print("theta type:", type(thetas_source))
                print("eta type:", type(etas_source))
                
                for i, theta in enumerate(thetas_source):
                    print(
                        f"theta {i}:",
                        type(theta),
                        theta.shape,
                    )
                
                for i, eta in enumerate(etas_source):
                    print(
                        f"eta {i}:",
                        type(eta),
                        eta.shape,
                    )
    
        # ======================================================
        # TARGET REPRESENTATION
        # ======================================================
    
        if target_images is not None:

            target_images = [
                image.to(
                    device,
                    non_blocking=True,
                )
                for image in target_images
            ]
        
            target_queries = []
            target_theta_values = []
        
            with torch.inference_mode():
                with torch.cuda.amp.autocast():
        
                    for target_image in target_images:
        
                        # calculate_theta returns:
                        # thetas, mus, logvars
                        target_thetas, _, _ = self.calculate_theta(
                            [target_image]
                        )
        
                        theta = target_thetas[0]
        
                        # calculate_eta returns a list
                        target_etas = self.calculate_eta(
                            [target_image]
                        )
        
                        eta = target_etas[0]
        
                        print(
                            "target theta:",
                            type(theta),
                            theta.shape,
                        )
        
                        print(
                            "target eta:",
                            type(eta),
                            eta.shape,
                        )
        
                        query = torch.cat(
                            [
                                theta,
                                eta,
                            ],
                            dim=1,
                        )
        
                        target_queries.append(
                            query
                        )
        
                        target_theta_values.append(
                            theta
                        )
            
        else:
    
            target_theta = target_theta.to(
                device
            )
    
            target_eta = target_eta.to(
                device
            )
    
            target_queries = [
                torch.cat(
                    [
                        target_theta[i:i + 1],
                        target_eta[i:i + 1],
                    ],
                    dim=1,
                )
                for i in range(
                    target_theta.shape[0]
                )
            ]
    
        # ======================================================
        # STACK SOURCE β LOGITS / KEYS
        # ======================================================
    
        # logits:
        # list N of [B,beta_dim,D,H,W]
        #
        # ->
        #
        # [B,N,beta_dim,D,H,W]
    
        logits_stack = torch.stack(
            logits,
            dim=1,
        )
    
        # [B,N,theta_dim+eta_dim]
    
        keys_stack = torch.stack(
            keys,
            dim=1,
        )
    
        print(
            "logits_stack:",
            logits_stack.shape,
        )
    
        print(
            "keys_stack:",
            keys_stack.shape,
        )
    
        # ======================================================
        # ALL SOURCE MODALITIES AVAILABLE
        # ======================================================
    
        B = logits_stack.shape[0]
        N = logits_stack.shape[1]
    
        modality_dropout = torch.zeros(
            (B, N),
            dtype=torch.bool,
            device=device,
        )
    
        # No spatial mask during inference for now.
        masks = None
    
        # ======================================================
        # RECONSTRUCT EACH TARGET
        # ======================================================
    
        for (
            target_index,
            query,
            out_path,
            norm_val,
        ) in zip(
            range(len(target_queries)),
            target_queries,
            out_paths,
            norm_vals,
        ):
    
            print()
            print(
                f"Harmonizing target {target_index + 1}/"
                f"{len(target_queries)}"
            )
    
            with torch.inference_mode():
    
                with torch.cuda.amp.autocast():
    
                    # --------------------------------------------
                    # Attention fusion
                    # --------------------------------------------
    
                    (
                        logit_fusion,
                        attention,
                    ) = self.attention_module(
                        q=query,
                        k=keys_stack,
                        v=logits_stack,
                        mask=masks,
                        modality_dropout=modality_dropout,
                    )
    
                    # --------------------------------------------
                    # Convert β logits to continuous anatomy map
                    # --------------------------------------------
    
                    beta_fusion = (
                        self.channel_aggregation(
                            reparameterize_logit(
                                logit_fusion
                            )
                        )
                    )
    
                    # --------------------------------------------
                    # Decoder expects:
                    #
                    # β + target θ
                    # --------------------------------------------
    
                    _, _, D, H, W = (
                        beta_fusion.shape
                    )
    
                    if target_images is not None:
                
                        target_theta_current = (
                            target_theta_values[
                                target_index
                            ]
                        )
                    
                    else:
                    
                        target_theta_current = (
                            target_theta[
                                target_index:
                                target_index + 1
                            ]
                        )
    
                    theta_map = (
                        target_theta_current[
                            :,
                            :,
                            None,
                            None,
                            None,
                        ]
                        .expand(
                            -1,
                            -1,
                            D,
                            H,
                            W,
                        )
                    )
    
                    decoder_input = torch.cat(
                        [
                            beta_fusion,
                            theta_map,
                        ],
                        dim=1,
                    )
    
                    rec_image = self.decoder(
                        decoder_input
                    )
    
            # ==================================================
            # SAVE OUTPUT
            # ==================================================
    
            rec_image = (
                rec_image
                .detach()
                .float()
                .cpu()
                .numpy()
            )
    
            # [1,1,D,H,W] -> [D,H,W]
            rec_image = rec_image[
                0,
                0,
            ]
    
            # Return to requested intensity scale
            rec_image = (
                rec_image
                * float(norm_val)
            )
    
            # Avoid negative MRI intensities
            rec_image = np.clip(
                rec_image,
                a_min=0.0,
                a_max=None,
            )
    
            output_header = (
                header.copy()
                if header is not None
                else None
            )
    
            output_obj = nib.Nifti1Image(
                rec_image,
                affine,
                header=output_header,
            )
    
            nib.save(
                output_obj,
                str(out_path),
            )
    
            print(
                f"Saved: {out_path}"
            )
    
            print(
                f"Output shape: {rec_image.shape}"
            )
    
            print(
                f"Output range: "
                f"{rec_image.min():.3f} - "
                f"{rec_image.max():.3f}"
            )
            # ==================================================
            # SAVE INTERMEDIATE SOURCE + ATTENTION VOLUMES
            # ==================================================
            
            if save_intermediate:
            
                intermediate_out_dir = Path(
                    intermediate_out_dir
                )
            
                intermediate_out_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )
            
            
                # ----------------------------------------------
                # SAVE SOURCE IMAGES
                # ----------------------------------------------
            
                for source_idx, source_image in enumerate(
                    source_images
                ):
            
                    source_np = (
                        source_image
                        .detach()
                        .float()
                        .cpu()
                        .numpy()[0, 0]
                    )
            
                    source_obj = nib.Nifti1Image(
                        source_np,
                        affine,
                        header=header.copy(),
                    )
            
                    source_path = (
                        intermediate_out_dir
                        / f"source_{source_idx}.nii.gz"
                    )
            
                    nib.save(
                        source_obj,
                        str(source_path),
                    )
            
                    print(
                        f"Saved source: {source_path}"
                    )
            
            
                # ----------------------------------------------
                # SAVE ATTENTION MAPS
                #
                # attention:
                # [B, N, D, H, W]
                # ----------------------------------------------
            
                attention_np = (
                    attention
                    .detach()
                    .float()
                    .cpu()
                    .numpy()[0]
                )
            
                for source_idx in range(
                    attention_np.shape[0]
                ):
            
                    attention_vol = (
                        attention_np[source_idx]
                    )
            
                    attention_obj = nib.Nifti1Image(
                        attention_vol,
                        affine,
                        header=header.copy(),
                    )
            
                    attention_path = (
                        intermediate_out_dir
                        / (
                            f"target_{target_index:02d}_"
                            f"attention_source_{source_idx}.nii.gz"
                        )
                    )
            
                    nib.save(
                        attention_obj,
                        str(attention_path),
                    )
            
                    print(
                        f"Saved attention: {attention_path}"
                    )

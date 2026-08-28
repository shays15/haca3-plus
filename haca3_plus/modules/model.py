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
            batch_id % 1 == 1
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

    def harmonize(self, source_images, target_images, target_theta, target_eta, out_paths,
                  recon_orientation, norm_vals, header=None, num_batches=4, save_intermediate=False, intermediate_out_dir=None):
        if out_paths is not None:
            for out_path in out_paths:
                mkdir_p(out_path.parent)
        if save_intermediate:
            mkdir_p(intermediate_out_dir)
        if out_paths is not None:
            prefix = out_paths[0].name.replace('.nii.gz', '')
        with torch.set_grad_enabled(False):
            self.beta_encoder.eval()
            self.theta_encoder.eval()
            self.eta_encoder.eval()
            self.decoder.eval()

            # === 1. CALCULATE BETA, THETA, ETA FROM SOURCE IMAGES ===
            logits, betas, keys, masks = [], [], [], []
            for source_image in source_images:
                source_image = source_image.unsqueeze(1)
                source_image_batches = divide_into_batches(source_image, num_batches)
                mask_tmp, logit_tmp, beta_tmp, key_tmp = [], [], [], []
                for source_image_batch in source_image_batches:
                    batch_size = source_image_batch.shape[0]
                    source_image_batch = source_image_batch.to(self.device)
                    #mask = (source_image_batch > 1e-6) * 1.0
                    mask = (source_image_batch > 1e-2) * 1.0
                    logit = self.beta_encoder(source_image_batch)
                    beta = self.channel_aggregation(reparameterize_logit(logit))
                    theta_source, _ = self.theta_encoder(source_image_batch)
                    eta_source = self.eta_encoder(source_image_batch).view(batch_size, self.eta_dim, 1, 1)
                    mask_tmp.append(mask)
                    logit_tmp.append(logit)
                    beta_tmp.append(beta)
                    key_tmp.append(torch.cat([theta_source, eta_source], dim=1))
                masks.append(torch.cat(mask_tmp, dim=0))
                logits.append(torch.cat(logit_tmp, dim=0))
                betas.append(torch.cat(beta_tmp, dim=0))
                keys.append(torch.cat(key_tmp, dim=0))

            # === 2. CALCULATE THETA, ETA FOR TARGET IMAGES (IF NEEDED) ===
            if target_theta is None:
                queries, thetas_target = [], []
                for target_image in target_images:
                    target_image = target_image.to(self.device).unsqueeze(1)
                    theta_target, _ = self.theta_encoder(target_image)
                    theta_target = theta_target.mean(dim=0, keepdim=True)
                    eta_target = self.eta_encoder(target_image).mean(dim=0, keepdim=True).view(1, self.eta_dim, 1, 1)
                    thetas_target.append(theta_target)
                    queries.append(
                        torch.cat([theta_target, eta_target], dim=1).view(1, self.theta_dim + self.eta_dim, 1))
                if save_intermediate:
                    # save theta and eta of target images
                    with open(intermediate_out_dir / f'{prefix}_targets.txt', 'w') as fp:
                        fp.write(','.join(['img'] + [f'theta{i}' for i in range(self.theta_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_query in enumerate([query.squeeze().cpu().numpy().tolist() for query in queries]):
                            fp.write(','.join([f'target{i}'] + ['%.6f' % val for val in img_query]) + '\n')
            else:
                queries, thetas_target = [], []
                for target_theta_tmp, target_eta_tmp in zip(target_theta, target_eta):
                    thetas_target.append(target_theta_tmp.view(1, self.theta_dim, 1, 1).to(self.device))
                    queries.append(torch.cat([target_theta_tmp.view(1, self.theta_dim, 1).to(self.device),
                                              target_eta_tmp.view(1, self.eta_dim, 1).to(self.device)], dim=1))

            # === 3. SAVE ENCODED VARIABLES (IF REQUESTED) ===
            if save_intermediate and header is not None:
                if recon_orientation == 'axial':
                    # 3a. source images
                    for i, source_img in enumerate(source_images):
                        img_save = source_img.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]
                        nib.Nifti1Image(img_save, None, header).to_filename(
                            intermediate_out_dir / f'{prefix}_source{i}.nii.gz'
                        )
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1)
                    print(beta.shape)
                    if len(beta.shape) > 4:
                        beta = beta.squeeze(1)
                    beta = beta.permute(1, 2, 0, 3).permute(1, 0, 2, 3).cpu().numpy()
                    img_save = nib.Nifti1Image(beta[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                    file_name = intermediate_out_dir / f'{prefix}_source_betas_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)
                    # 3c. theta/eta values
                    with open(intermediate_out_dir / f'{prefix}_sources.txt', 'w') as fp:
                        fp.write(','.join(['img', 'slice'] + [f'theta{i}' for i in range(self.theta_dim)] +
                                          [f'eta{i}' for i in range(self.eta_dim)]) + '\n')
                        for i, img_key in enumerate([key.squeeze().cpu().numpy().tolist() for key in keys]):
                            for j, slice_key in enumerate(img_key):
                                fp.write(','.join([f'source{i}', f'slice{j:03d}'] +
                                                  ['%.6f' % val for val in slice_key]) + '\n')
                elif recon_orientation == 'coronal':
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1)
                    if len(beta.shape) > 4:
                        beta = beta.squeeze(1)
                    beta = beta.permute(1, 2, 0, 3).permute(1, 0, 2, 3).cpu().numpy()
                    img_save = nib.Nifti1Image(beta[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                    file_name = intermediate_out_dir / f'{prefix}_source_betas_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)
                elif recon_orientation == 'sagittal':
                    # 3b. beta images
                    beta = torch.stack(betas, dim=-1)
                    if len(beta.shape) > 4:
                        beta = beta.squeeze(1)
                    beta = beta.permute(1, 2, 0, 3).permute(1, 0, 2, 3).cpu().numpy()
                    img_save = nib.Nifti1Image(beta[:, :, 112 - 96:112 + 96, :], None, header)
                    file_name = intermediate_out_dir / f'{prefix}_source_betas_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)

            # ===4. DECODING===
            for tid, (theta_target, query, norm_val) in enumerate(zip(thetas_target, queries, norm_vals)):
                if out_paths is not None:
                    out_prefix = out_paths[tid].name.replace('.nii.gz', '')
                rec_image, beta_fusion, logit_fusion, attention = [], [], [], []
                for batch_id in range(num_batches):
                    keys_tmp = [divide_into_batches(ks, num_batches)[batch_id] for ks in keys]
                    logits_tmp = [divide_into_batches(ls, num_batches)[batch_id] for ls in logits]
                    masks_tmp = [divide_into_batches(ms, num_batches)[batch_id] for ms in masks]
                    batch_size = keys_tmp[0].shape[0]
                    query_tmp = query.view(1, self.theta_dim + self.eta_dim, 1).repeat(batch_size, 1, 1)
                    k = torch.cat(keys_tmp, dim=-1).view(batch_size, self.theta_dim + self.eta_dim, 1, len(source_images))
                    v = torch.stack(logits_tmp, dim=-1).view(batch_size, self.beta_dim, 224 * 224, len(source_images))
                    
                    #expanded_mask = masks_tmp[0].unsqueeze(1)
                    #expanded_mask = masks_tmp.expand(-1, attention.size(1), -1, -1, -1).squeeze(2)

                    
                    logit_fusion_tmp, attention_tmp = self.attention_module(query_tmp, k, v, masks_tmp, None, 5.0)
                    beta_fusion_tmp = self.channel_aggregation(reparameterize_logit(logit_fusion_tmp))
                    combined_map = torch.cat([beta_fusion_tmp, theta_target.repeat(batch_size, 1, 224, 224)], dim=1)
                    masks_cpu = [mask.cpu().numpy() for mask in masks_tmp]
                    union_mask = np.logical_or.reduce(masks_cpu)
                    union_mask = torch.from_numpy(union_mask).to(masks_tmp[0].device)
                    rec_image_tmp = self.decoder(combined_map) * union_mask
                    rec_image.append(rec_image_tmp)
                    beta_fusion.append(beta_fusion_tmp)
                    logit_fusion.append(logit_fusion_tmp)
                    attention.append(attention_tmp)

                rec_image = torch.cat(rec_image, dim=0)
                beta_fusion = torch.cat(beta_fusion, dim=0)
                logit_fusion = torch.cat(logit_fusion, dim=0)
                attention = torch.cat(attention, dim=0)

                # ===5. SAVE INTERMEDIATE RESULTS (IF REQUESTED)===
                # harmonized image
                if header is not None:
                    if recon_orientation == "axial":
                        img_save = np.array(rec_image.cpu().squeeze().permute(1, 2, 0).permute(1, 0, 2))
                    elif recon_orientation == "coronal":
                        img_save = np.array(rec_image.cpu().squeeze().permute(0, 2, 1).flip(2).permute(1, 0, 2))
                    else:
                        img_save = np.array(rec_image.cpu().squeeze().permute(2, 0, 1).flip(2).permute(1, 0, 2))
                    img_save = nib.Nifti1Image((img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96]) * norm_val, None,
                                               header)
                    file_name = out_path.parent / f'{out_prefix}_harmonized_{recon_orientation}.nii.gz'
                    nib.save(img_save, file_name)

                if save_intermediate and header is not None:
                    # 5a. beta fusion
                    if recon_orientation == 'axial':
                        img_save = beta_fusion.squeeze().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_beta_fusion.nii.gz'
                        nib.save(img_save, file_name)
                    # 5b. logit fusion
                    if recon_orientation == 'axial':
                        img_save = logit_fusion.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96, :], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_logit_fusion.nii.gz'
                        nib.save(img_save, file_name)
                    # 5c. attention
                    if recon_orientation == 'axial':
                        img_save = attention.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                        img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                        file_name = intermediate_out_dir / f'{out_prefix}_attention.nii.gz'
                        nib.save(img_save, file_name)
                    # # 5d. attention_map
                    # if recon_orientation == 'axial' and attention_map != []:
                    #     img_save = attention_map.permute(2, 3, 0, 1).permute(1, 0, 2, 3).cpu().numpy()
                    #     img_save = nib.Nifti1Image(img_save[112 - 96:112 + 96, :, 112 - 96:112 + 96], None, header)
                    #     file_name = intermediate_out_dir / f'{out_prefix}_attention_map.nii.gz'
                    #     nib.save(img_save, file_name)
        if header is None:
            return rec_image.cpu().squeeze()

    def combine_images(self, image_paths, out_path, norm_val, pretrained_fusion=None):
        # obtain images
        images = []
        for image_path in image_paths:
            image_pad = torch.zeros((224, 224, 224))
            image_obj = nib.load(image_path)
            image_vol, _ = normalize_intensity(torch.from_numpy(image_obj.get_fdata().astype(np.float32)))
            image_pad[112 - 96:112 + 96, :, 112 - 96:112 + 96] = image_vol
            image_header = image_obj.header
            images.append(image_pad.numpy())

        if pretrained_fusion is not None:
            checkpoint = torch.load(pretrained_fusion, map_location=self.device)
            fusion_net = FusionNet(in_ch=3, out_ch=1)
            fusion_net.load_state_dict(checkpoint['fusion_net'])
            fusion_net.to(self.device)
            fusion_net.eval()
            with autocast():
                image = torch.cat(
                    [ToTensor()(im).permute(2, 1, 0).permute(2, 0, 1).unsqueeze(0).unsqueeze(0) for im in images],
                    dim=1).to(self.device)
                image_fusion = fusion_net(image).squeeze().detach().permute(1, 2, 0).permute(1, 0, 2).cpu().numpy()
        else:
            # calculate median
            image_cat = np.stack(images, axis=-1)
            image_fusion = np.median(image_cat, axis=-1)

        # save fusion_image
        img_save = image_fusion[112 - 96:112 + 96, :, 112 - 96:112 + 96] * norm_val
        img_save = nib.Nifti1Image(img_save, None, image_header)
        prefix = out_path.name.replace('.nii.gz', '')
        file_name = out_path.parent / f'{prefix}_harmonized_fusion.nii.gz'
        nib.save(img_save, file_name)

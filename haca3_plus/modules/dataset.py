import os
from glob import glob

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import torchio as tio
import nibabel as nib


# ==========================================================
# SETTINGS
# ==========================================================

contrast_names = [
    "T1PRE",
    "T2",
    "PD",
    "FLAIR",
]

expected_shape = (192, 224, 192)


# ==========================================================
# DEGRADATION / ARTIFACT AUGMENTATION
# ==========================================================

transform_dict = {
    tio.RandomMotion(
        degrees=(15, 30),
        translation=(10, 20)
    ): 0.25,

    tio.RandomNoise(
        std=(0.01, 0.1)
    ): 0.25,

    tio.RandomGhosting(
        num_ghosts=(4, 10)
    ): 0.25,

    tio.RandomBiasField(): 0.25,
}

degradation_transform = tio.OneOf(
    transform_dict
)


# ==========================================================
# IMAGE LOADING
# ==========================================================

def get_tensor_from_fpath(
    fpath,
    normalization_method
):
    """
    Load one full 3D NIfTI.

    Returns
    -------
    image : torch.Tensor
        Shape [1, D, H, W]

    exists : int
        1 if image exists, 0 otherwise
    """

    # ------------------------------------------------------
    # Missing contrast
    # ------------------------------------------------------

    if not os.path.exists(fpath):

        image = torch.zeros(
            [1, *expected_shape],
            dtype=torch.float32
        )

        exists = 0

        return image, exists


    # ------------------------------------------------------
    # Load NIfTI
    # ------------------------------------------------------

    image = nib.load(
        fpath
    ).get_fdata(
        dtype=np.float32
    )

    image = np.squeeze(image)


    if image.ndim != 3:
        raise ValueError(
            f"Expected 3D image for {fpath}, "
            f"got shape {image.shape}"
        )


    # ------------------------------------------------------
    # Verify registered dimensions
    # ------------------------------------------------------

    if image.shape != expected_shape:
        raise ValueError(
            f"Unexpected shape for:\n"
            f"{fpath}\n"
            f"Expected {expected_shape}, "
            f"got {image.shape}"
        )


    # ------------------------------------------------------
    # Normalize
    #
    # Keep the original HACA3 normalization logic for now.
    # np.percentile works exactly the same on a 3D volume.
    # ------------------------------------------------------

    if normalization_method == "wm":

        image = image / 2.0


    elif normalization_method == "01":

        p95 = np.percentile(
            image,
            95
        )

        image = image / (
            p95 + 1e-5
        )

        image = np.clip(
            image,
            0.0,
            5.0
        )


    # ------------------------------------------------------
    # numpy -> torch
    #
    # [D,H,W] -> [1,D,H,W]
    # ------------------------------------------------------

    image = np.ascontiguousarray(
        image
    )

    image = torch.from_numpy(
        image
    ).unsqueeze(0)

    exists = 1


    return image, exists


# ==========================================================
# COMMON BRAIN MASK
# ==========================================================

def background_removal(
    image_dicts
):
    """
    Construct a common foreground mask using only contrasts
    that actually exist.

    All images:
        [1,D,H,W]

    mask:
        [1,D,H,W]
    """

    existing_images = [
        image_dict["image"]
        for image_dict in image_dicts
        if image_dict["exists"] == 1
    ]


    if len(existing_images) == 0:
        raise RuntimeError(
            "Subject has no available contrasts."
        )


    # Start with all foreground
    mask = torch.ones_like(
        existing_images[0],
        dtype=torch.bool
    )


    # Intersection of foreground across AVAILABLE contrasts
    for image in existing_images:

        mask = (
            mask
            & image.ge(1e-8)
        )


    # Apply same mask to all contrasts
    for image_dict in image_dicts:

        image_dict["image"] = (
            image_dict["image"]
            * mask
        )

        image_dict["image_degrade"] = (
            image_dict["image_degrade"]
            * mask
        )

        image_dict["mask"] = mask


    return image_dicts


# ==========================================================
# DATASET
# ==========================================================

class HACA3Dataset(Dataset):

    def __init__(
        self,
        dataset_dirs,
        contrasts,
        mode="train",
        normalization_method="01"
    ):

        self.mode = mode

        self.dataset_dirs = (
            dataset_dirs
        )

        self.contrasts = (
            contrasts
        )

        self.normalization_method = (
            normalization_method
        )

        (
            self.t1_paths,
            self.site_ids
        ) = self._get_file_paths()


    # ======================================================
    # FIND SUBJECT / SESSION VOLUMES
    # ======================================================

    def _get_file_paths(self):

        fpaths = []
        site_ids = []


        for site_id, dataset_dir in enumerate(
            self.dataset_dirs
        ):

            t1_paths = sorted(
                glob(
                    os.path.join(
                        dataset_dir,
                        self.mode,
                        "*T1PRE*.nii.gz"
                    )
                )
            )


            fpaths += t1_paths

            site_ids += (
                [site_id]
                * len(t1_paths)
            )


        return (
            fpaths,
            site_ids
        )


    # ======================================================
    # LENGTH
    # ======================================================

    def __len__(self):

        return len(
            self.t1_paths
        )


    # ======================================================
    # GET ONE SUBJECT / SESSION
    # ======================================================

    def __getitem__(
        self,
        idx: int
    ):

        image_dicts = []

        t1_path = (
            self.t1_paths[idx]
        )

        site_id = (
            self.site_ids[idx]
        )


        for (
            contrast_id,
            contrast_name
        ) in enumerate(
            contrast_names
        ):

            # ----------------------------------------------
            # Find corresponding contrast
            # ----------------------------------------------

            image_path = (
                t1_path.replace(
                    "T1PRE",
                    contrast_name
                )
            )


            # ----------------------------------------------
            # Load full 3D volume
            # ----------------------------------------------

            image, exists = (
                get_tensor_from_fpath(
                    image_path,
                    self.normalization_method
                )
            )


            # ----------------------------------------------
            # Apply 3D degradation
            #
            # image is already [C,D,H,W].
            # TorchIO can operate directly on this.
            # ----------------------------------------------

            if (
                self.mode == "train"
                and exists
            ):

                image_degrade = (
                    degradation_transform(
                        image
                    )
                )

            else:

                image_degrade = (
                    image.clone()
                )


            # ----------------------------------------------
            # Store metadata
            # ----------------------------------------------

            image_dict = {
                "image": image,

                "image_degrade":
                    image_degrade,

                "site_id":
                    site_id,

                "contrast_id":
                    contrast_id,

                "exists":
                    exists,

                "path":
                    image_path,
            }


            image_dicts.append(
                image_dict
            )


        # ----------------------------------------------
        # Common foreground
        # ----------------------------------------------

        image_dicts = (
            background_removal(
                image_dicts
            )
        )


        return image_dicts

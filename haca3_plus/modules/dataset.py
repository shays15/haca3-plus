import os
from glob import glob
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torchio as tio
from torch.utils.data.dataset import Dataset


# ==========================================================
# SETTINGS
# ==========================================================

contrast_names = [
    "T1PRE",
    "T2",
    "PD",
    "FLAIR",
]

# Map the HACA3 contrast names to patterns in the RADIFOX /
# TREAT-MS filenames.
contrast_patterns = {
    "T1PRE": "*BRAIN-T1-*PRE*",
    "T2": "*BRAIN-T2-*",
    "PD": "*BRAIN-PD-*",
    "FLAIR": "*BRAIN-FLAIR-*",
}

expected_shape = (
    192,
    224,
    192,
)


# ==========================================================
# DEGRADATION / ARTIFACT AUGMENTATION
# ==========================================================

transform_dict = {
    tio.RandomMotion(
        degrees=(15, 30),
        translation=(10, 20),
    ): 0.25,

    tio.RandomNoise(
        std=(0.01, 0.1),
    ): 0.25,

    tio.RandomGhosting(
        num_ghosts=(4, 10),
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
    normalization_method,
):
    """
    Load one full 3D NIfTI.

    Parameters
    ----------
    fpath
        Path to NIfTI. Can be None if contrast is missing.

    normalization_method
        "wm", "01", or other/no normalization.

    Returns
    -------
    image : torch.Tensor
        Shape [1, D, H, W]

    exists : int
        1 if image exists, otherwise 0.
    """

    # ------------------------------------------------------
    # Missing contrast
    # ------------------------------------------------------

    if fpath is None:

        image = torch.zeros(
            [1, *expected_shape],
            dtype=torch.float32,
        )

        return image, 0


    fpath = Path(fpath)


    if not fpath.exists():

        image = torch.zeros(
            [1, *expected_shape],
            dtype=torch.float32,
        )

        return image, 0


    # ------------------------------------------------------
    # Load image
    # ------------------------------------------------------

    image = nib.load(
        str(fpath)
    ).get_fdata(
        dtype=np.float32
    )

    image = np.squeeze(
        image
    )


    if image.ndim != 3:

        raise ValueError(
            f"Expected 3D image for:\n"
            f"{fpath}\n"
            f"Got shape {image.shape}"
        )


    # ------------------------------------------------------
    # Verify dimensions
    # ------------------------------------------------------

    if image.shape != expected_shape:

        raise ValueError(
            f"Unexpected image shape for:\n"
            f"{fpath}\n"
            f"Expected: {expected_shape}\n"
            f"Got:      {image.shape}"
        )


    # ------------------------------------------------------
    # Normalize
    #
    # This preserves the original HACA3 normalization logic.
    # np.percentile will flatten the full 3D image.
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
            5.0,
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

    return image, 1


# ==========================================================
# COMMON FOREGROUND MASK
# ==========================================================

def background_removal(
    image_dicts,
):
    """
    Construct a common foreground mask from contrasts that
    actually exist.

    Each image has shape:
        [1,D,H,W]

    The returned mask has shape:
        [1,D,H,W]
    """

    existing_images = [
        image_dict["image"]
        for image_dict in image_dicts
        if image_dict["exists"] == 1
    ]


    if len(existing_images) == 0:

        raise RuntimeError(
            "No available contrasts for this sample."
        )


    # ------------------------------------------------------
    # Begin with all foreground
    # ------------------------------------------------------

    mask = torch.ones_like(
        existing_images[0],
        dtype=torch.bool,
    )


    # ------------------------------------------------------
    # Intersection of foreground across existing contrasts
    # ------------------------------------------------------

    for image in existing_images:

        mask = (
            mask
            & image.ge(1e-8)
        )


    # ------------------------------------------------------
    # Apply common mask
    # ------------------------------------------------------

    for image_dict in image_dicts:

        image_dict["image"] = (
            image_dict["image"]
            * mask
        )

        image_dict["image_degrade"] = (
            image_dict["image_degrade"]
            * mask
        )

        image_dict["mask"] = (
            mask
        )


    return image_dicts


# ==========================================================
# DATASET
# ==========================================================

class HACA3Dataset(Dataset):

    def __init__(
        self,
        dataset_dirs,
        contrasts=None,
        mode="train",
        normalization_method="01",
    ):

        self.mode = mode

        self.dataset_dirs = [
            Path(dataset_dir)
            for dataset_dir in dataset_dirs
        ]

        self.contrasts = (
            contrasts
            if contrasts is not None
            else contrast_names
        )

        self.normalization_method = (
            normalization_method
        )


        (
            self.t1_paths,
            self.site_ids,
        ) = self._get_file_paths()


        print(
            f"HACA3Dataset ({self.mode}): "
            f"{len(self.t1_paths)} samples"
        )


    # ======================================================
    # FIND T1 ANCHORS
    # ======================================================

    def _get_file_paths(
        self,
    ):

        fpaths = []
        site_ids = []


        for (
            site_id,
            dataset_dir,
        ) in enumerate(
            self.dataset_dirs
        ):

            # --------------------------------------------------
            # TREAT-MS directories contain the NIfTIs directly.
            #
            # Example:
            #
            # TREATMS-0100-006_20190912_01-03_
            # BRAIN-T1-IRFSPGR-3D-SAGITTAL-PRE_...
            # --------------------------------------------------

            t1_paths = sorted(
                dataset_dir.glob(
                    "*BRAIN-T1-*PRE*.nii.gz"
                )
            )


            print(
                f"{dataset_dir}: "
                f"found {len(t1_paths)} T1 PRE volumes"
            )


            fpaths.extend(
                t1_paths
            )

            site_ids.extend(
                [site_id]
                * len(t1_paths)
            )


        return (
            fpaths,
            site_ids,
        )


    # ======================================================
    # LENGTH
    # ======================================================

    def __len__(
        self,
    ):

        return len(
            self.t1_paths
        )


    # ======================================================
    # FIND ONE CONTRAST
    # ======================================================

    def _find_contrast(
        self,
        t1_path,
        contrast_name,
    ):
        """
        Find the corresponding contrast for the same
        subject/session as the T1 anchor.

        Example T1 filename:

        TREATMS-0100-006_20190912_01-03_
        BRAIN-T1-IRFSPGR-3D-SAGITTAL-PRE_...

        Subject/session prefix becomes:

        TREATMS-0100-006_20190912
        """

        t1_path = Path(
            t1_path
        )


        # --------------------------------------------------
        # Get subject/session prefix
        # --------------------------------------------------

        parts = (
            t1_path.name.split("_")
        )

        if len(parts) < 2:

            raise ValueError(
                f"Could not extract subject/session "
                f"from filename:\n"
                f"{t1_path.name}"
            )


        subj_sess = "_".join(
            parts[:2]
        )


        # --------------------------------------------------
        # Get pattern for requested contrast
        # --------------------------------------------------

        contrast_pattern = (
            contrast_patterns[
                contrast_name
            ]
        )


        pattern = (
            f"{subj_sess}_"
            f"{contrast_pattern}"
            f".nii.gz"
        )


        matches = sorted(
            t1_path.parent.glob(
                pattern
            )
        )


        # --------------------------------------------------
        # No image
        # --------------------------------------------------

        if len(matches) == 0:

            return None


        # --------------------------------------------------
        # Exactly one
        # --------------------------------------------------

        if len(matches) == 1:

            return matches[0]


        # --------------------------------------------------
        # Multiple candidate scans
        #
        # Do NOT silently ignore this yet.
        # For now print them and select the first.
        # --------------------------------------------------

        print()

        print(
            f"[WARNING] "
            f"{subj_sess}: "
            f"found {len(matches)} "
            f"{contrast_name} images"
        )


        for match in matches:

            print(
                f"    {match.name}"
            )


        print(
            f"    -> temporarily using: "
            f"{matches[0].name}"
        )

        print()


        return matches[0]


    # ======================================================
    # GET ONE SAMPLE
    # ======================================================

    def __getitem__(
        self,
        idx: int,
    ):

        image_dicts = []


        # --------------------------------------------------
        # Anchor T1
        # --------------------------------------------------

        t1_path = Path(
            self.t1_paths[idx]
        )

        site_id = (
            self.site_ids[idx]
        )


        # --------------------------------------------------
        # Extract subject/session name
        # --------------------------------------------------

        parts = (
            t1_path.name.split("_")
        )

        subj_sess = "_".join(
            parts[:2]
        )


        # --------------------------------------------------
        # Load each requested contrast
        # --------------------------------------------------

        for (
            contrast_id,
            contrast_name,
        ) in enumerate(
            self.contrasts
        ):

            # ----------------------------------------------
            # Find matching contrast
            # ----------------------------------------------

            image_path = (
                self._find_contrast(
                    t1_path,
                    contrast_name,
                )
            )


            # ----------------------------------------------
            # Load volume
            # ----------------------------------------------

            (
                image,
                exists,
            ) = get_tensor_from_fpath(
                image_path,
                self.normalization_method,
            )


            # ----------------------------------------------
            # Apply 3D artifact degradation
            #
            # TorchIO input:
            # [C,D,H,W]
            # ----------------------------------------------

            if (
                self.mode == "train"
                and exists == 1
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
            # Store data
            # ----------------------------------------------

            image_dict = {

                "image":
                    image,

                "image_degrade":
                    image_degrade,

                "site_id":
                    site_id,

                "contrast_id":
                    contrast_id,

                "contrast_name":
                    contrast_name,

                "exists":
                    exists,

                "path":
                    (
                        str(image_path)
                        if image_path is not None
                        else ""
                    ),

                "subj_sess":
                    subj_sess,
            }


            image_dicts.append(
                image_dict
            )


        # --------------------------------------------------
        # Common foreground mask
        # --------------------------------------------------

        image_dicts = (
            background_removal(
                image_dicts
            )
        )


        return image_dicts

import sys
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from skimage.filters import threshold_otsu
from skimage.morphology import isotropic_closing

from .modules.model import HACA3
from .modules.utils import mkdir_p


# ==========================================================
# BACKGROUND REMOVAL
# ==========================================================

def background_removal(image_vol):
    """
    Optional 3D foreground masking.

    image_vol:
        [D, H, W]
    """

    if np.max(image_vol) <= 0:
        return image_vol

    thresh = threshold_otsu(image_vol)

    mask = image_vol >= thresh

    # 3D morphological closing
    mask = isotropic_closing(
        mask,
        radius=20,
    )

    image_vol = image_vol.copy()
    image_vol[~mask] = 0.0

    return image_vol


# ==========================================================
# LOAD AND NORMALIZE A SINGLE 3D IMAGE
# ==========================================================

def obtain_single_image(
    image_path,
    bg_removal=True,
    normalization_method="01",
):
    """
    Load a full 3D NIfTI volume.

    Returns
    -------
    image_tensor:
        [1, 1, D, H, W]

    affine:
        Original NIfTI affine.

    header:
        Original NIfTI header.

    norm_val:
        Normalization value used for eventual conversion
        back to approximately the original intensity scale.
    """

    image_obj = nib.load(str(image_path))

    image_vol = image_obj.get_fdata(
        dtype=np.float32
    )


    # ------------------------------------------------------
    # Normalization
    # ------------------------------------------------------

    if normalization_method == "01":

        norm_val = np.percentile(
            image_vol,
            95,
        )

        image_vol = (
            image_vol /
            (norm_val + 1e-5)
        )

        image_vol = np.clip(
            image_vol,
            0.0,
            5.0,
        )

    elif normalization_method == "wm":

        norm_val = 2.0

        image_vol = (
            image_vol / 2.0
        )

    else:

        norm_val = 1.0


    # ------------------------------------------------------
    # Optional foreground removal
    # ------------------------------------------------------

    if bg_removal:

        image_vol = background_removal(
            image_vol
        )


    # ------------------------------------------------------
    # NumPy -> Torch
    #
    # [D,H,W]
    #     ->
    # [1,1,D,H,W]
    # ------------------------------------------------------

    image_tensor = torch.from_numpy(
        image_vol
    ).float()

    image_tensor = (
        image_tensor
        .unsqueeze(0)
        .unsqueeze(0)
    )


    return (
        image_tensor,
        image_obj.affine,
        image_obj.header.copy(),
        norm_val,
    )


# ==========================================================
# LOAD SOURCE IMAGES
# ==========================================================

def load_source_images(
    image_paths,
    bg_removal=True,
    normalization_method="01",
):
    """
    Load all source contrasts.

    Each returned image:
        [1,1,D,H,W]
    """

    source_images = []

    reference_affine = None
    reference_header = None


    for image_path in image_paths:

        (
            image,
            affine,
            header,
            _,
        ) = obtain_single_image(
            image_path,
            bg_removal=bg_removal,
            normalization_method=normalization_method,
        )


        if reference_affine is None:

            reference_affine = affine
            reference_header = header


        source_images.append(
            image
        )


    # ------------------------------------------------------
    # Ensure all registered volumes have the same dimensions
    # ------------------------------------------------------

    shapes = [
        tuple(image.shape)
        for image in source_images
    ]


    if len(set(shapes)) != 1:

        raise ValueError(
            "All source images must have identical dimensions. "
            f"Found: {shapes}"
        )


    return (
        source_images,
        reference_affine,
        reference_header,
    )


# ==========================================================
# MAIN
# ==========================================================

def main(args=None):

    args = (
        sys.argv[1:]
        if args is None
        else args
    )


    parser = argparse.ArgumentParser(
        description="3D Harmonization with HACA3+."
    )


    # ------------------------------------------------------
    # INPUTS
    # ------------------------------------------------------

    parser.add_argument(
        "--in-path",
        type=Path,
        action="append",
        required=True,
        help=(
            "Source registered MRI volume. "
            "Specify once per available contrast."
        ),
    )


    # ------------------------------------------------------
    # TARGET CONTRAST
    # ------------------------------------------------------

    parser.add_argument(
        "--target-image",
        type=Path,
        action="append",
        default=[],
    )

    parser.add_argument(
        "--target-theta",
        type=float,
        nargs=2,
        action="append",
        default=[],
    )

    parser.add_argument(
        "--target-eta",
        type=float,
        nargs=2,
        action="append",
        default=[],
    )


    # ------------------------------------------------------
    # INTENSITY NORMALIZATION
    # ------------------------------------------------------

    parser.add_argument(
        "--norm-val",
        type=float,
        action="append",
        default=[],
    )

    parser.add_argument(
        "--normalization-method",
        type=str,
        default="01",
        choices=[
            "01",
            "wm",
            "none",
        ],
    )


    # ------------------------------------------------------
    # OUTPUT
    # ------------------------------------------------------

    parser.add_argument(
        "--out-path",
        type=Path,
        action="append",
        required=True,
    )


    # ------------------------------------------------------
    # MODEL
    # ------------------------------------------------------

    parser.add_argument(
        "--harmonization-model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--beta-dim",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--theta-dim",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--eta-dim",
        type=int,
        default=2,
    )


    # ------------------------------------------------------
    # OPTIONAL OUTPUTS
    # ------------------------------------------------------

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--intermediate-out-dir",
        type=Path,
        default=Path.cwd(),
    )


    # ------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------

    parser.add_argument(
        "--no-bg-removal",
        dest="bg_removal",
        action="store_false",
        default=True,
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )


    args = parser.parse_args(args)


    # ======================================================
    # BEGIN
    # ======================================================

    text_div = "=" * 10

    print(
        f"{text_div} BEGIN 3D HACA3+ HARMONIZATION {text_div}"
    )


    # ======================================================
    # ABSOLUTE PATHS
    # ======================================================

    args.in_path = [
        path.resolve()
        for path in args.in_path
    ]

    args.target_image = [
        path.resolve()
        for path in args.target_image
    ]

    args.out_path = [
        path.resolve()
        for path in args.out_path
    ]

    args.harmonization_model = (
        args.harmonization_model.resolve()
    )

    args.intermediate_out_dir = (
        args.intermediate_out_dir.resolve()
    )


    # ======================================================
    # ARGUMENT CHECKS
    # ======================================================

    using_target_image = (
        len(args.target_image) > 0
    )

    using_target_theta = (
        len(args.target_theta) > 0
    )


    if not (
        using_target_image
        ^ using_target_theta
    ):

        parser.error(
            "Provide either --target-image "
            "or --target-theta."
        )


    if (
        using_target_image
        and
        len(args.target_image)
        != len(args.out_path)
    ):

        parser.error(
            "Number of --target-image and "
            "--out-path arguments must match."
        )


    # ======================================================
    # DEFAULT TARGET ETA
    # ======================================================

    if (
        using_target_theta
        and
        len(args.target_eta) == 0
    ):

        args.target_eta = [
            [0.3, 0.5]
        ]


    # ======================================================
    # BROADCAST TARGET THETA / ETA
    # ======================================================

    if (
        len(args.target_theta) == 1
        and
        len(args.target_eta) > 1
    ):

        args.target_theta = (
            args.target_theta
            * len(args.target_eta)
        )


    if (
        len(args.target_theta) > 1
        and
        len(args.target_eta) == 1
    ):

        args.target_eta = (
            args.target_eta
            * len(args.target_theta)
        )


    if (
        using_target_theta
        and
        len(args.target_theta)
        != len(args.out_path)
    ):

        parser.error(
            "Number of --target-theta and "
            "--out-path arguments must match."
        )


    if (
        len(args.target_theta) > 0
        and
        len(args.target_theta)
        != len(args.target_eta)
    ):

        parser.error(
            "Number of --target-theta and "
            "--target-eta arguments must match."
        )


    # ======================================================
    # INTERMEDIATE DIRECTORY
    # ======================================================

    if args.save_intermediate:

        mkdir_p(
            args.intermediate_out_dir
        )


    # ======================================================
    # INITIALIZE MODEL
    # ======================================================

    print()
    print("Loading HACA3+ model...")

    haca3 = HACA3(
        beta_dim=args.beta_dim,
        theta_dim=args.theta_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.harmonization_model,
        gpu_id=args.gpu_id,
    )


    # ======================================================
    # LOAD SOURCE IMAGES
    # ======================================================

    print()
    print("Loading source images...")


    (
        source_images,
        image_affine,
        image_header,
    ) = load_source_images(
        args.in_path,
        bg_removal=args.bg_removal,
        normalization_method=args.normalization_method,
    )


    print(
        "Number of source images:",
        len(source_images),
    )


    for i, image in enumerate(
        source_images
    ):

        print(
            f"source {i}: "
            f"shape={tuple(image.shape)}, "
            f"min={image.min().item():.4f}, "
            f"max={image.max().item():.4f}"
        )


    # ======================================================
    # LOAD TARGET IMAGE(S)
    # ======================================================

    if using_target_image:

        target_images = []
        norm_vals = []


        for target_path in args.target_image:

            (
                target_image,
                _,
                _,
                norm_val,
            ) = obtain_single_image(
                target_path,
                bg_removal=args.bg_removal,
                normalization_method=args.normalization_method,
            )


            target_images.append(
                target_image
            )

            norm_vals.append(
                norm_val
            )


        target_theta = None
        target_eta = None


    # ======================================================
    # OR USE PROVIDED THETA / ETA
    # ======================================================

    else:

        target_images = None


        target_theta = torch.as_tensor(
            args.target_theta,
            dtype=torch.float32,
        )


        target_eta = torch.as_tensor(
            args.target_eta,
            dtype=torch.float32,
        )


        if len(args.norm_val) == 0:

            norm_vals = [
                1000.0
            ] * len(
                args.target_theta
            )

        elif (
            len(args.norm_val) == 1
            and
            len(args.target_theta) > 1
        ):

            norm_vals = (
                args.norm_val
                * len(args.target_theta)
            )

        else:

            norm_vals = args.norm_val


    # ======================================================
    # 3D HARMONIZATION
    # ======================================================

    print()
    print(
        f"{text_div} START 3D HARMONIZATION {text_div}"
    )


    haca3.harmonize(
        source_images=source_images,
        target_images=target_images,
        target_theta=target_theta,
        target_eta=target_eta,
        out_paths=args.out_path,
        affine=image_affine,
        header=image_header,
        norm_vals=norm_vals,
        save_intermediate=args.save_intermediate,
        intermediate_out_dir=args.intermediate_out_dir,
    )


    print()
    print(
        f"{text_div} COMPLETE {text_div}"
    )


if __name__ == "__main__":
    main()

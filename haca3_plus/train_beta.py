import argparse
import sys

from .modules.beta_model import BetaModel


def main(args=None):

    args = sys.argv[1:] if args is None else args

    parser = argparse.ArgumentParser(
        description="Pretrain the 3D HACA3+ beta encoder."
    )


    # ======================================================
    # DATA
    # ======================================================

    parser.add_argument(
        "--dataset-dirs",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Dataset/site directories containing "
            "train/valid folders."
        ),
    )

    parser.add_argument(
        "--contrasts",
        type=str,
        nargs="+",
        required=True,
        help="Contrasts to use, e.g. T1PRE T2 PD FLAIR.",
    )

    parser.add_argument(
        "--normalization-method",
        type=str,
        default="01",
        choices=["01", "wm", "none"],
    )


    # ======================================================
    # OUTPUT
    # ======================================================

    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
    )


    # ======================================================
    # MODEL
    # ======================================================

    parser.add_argument(
        "--beta-dim",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help=(
            "Patch size used to construct spatial beta "
            "features for contrastive learning."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for beta contrastive loss.",
    )


    # ======================================================
    # TRAINING
    # ======================================================

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Number of subject/sessions per batch. "
            "Batch size 1 is valid for beta training because "
            "spatial locations provide the negatives."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
    )


    # ======================================================
    # GPU
    # ======================================================

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
    )


    args = parser.parse_args(args)


    # ======================================================
    # PRINT CONFIGURATION
    # ======================================================

    text_div = "=" * 10

    print(
        f"{text_div} BEGIN BETA ENCODER TRAINING {text_div}"
    )

    print()

    print("Dataset dirs:")

    for dataset_dir in args.dataset_dirs:
        print(f"    {dataset_dir}")

    print()

    print(
        "Contrasts:",
        args.contrasts,
    )

    print(
        "Beta dim:",
        args.beta_dim,
    )

    print(
        "Patch size:",
        args.patch_size,
    )

    print(
        "Temperature:",
        args.temperature,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    print(
        "Learning rate:",
        args.lr,
    )

    print(
        "Epochs:",
        args.epochs,
    )

    print(
        "GPU:",
        args.gpu_id,
    )

    print()


    # ======================================================
    # 1. INITIALIZE MODEL
    # ======================================================

    beta_model = BetaModel(
        beta_dim=args.beta_dim,
        gpu_id=args.gpu_id,
        temperature=args.temperature,
        patch_size=args.patch_size,
    )


    # ======================================================
    # 2. LOAD DATASETS
    # ======================================================

    beta_model.load_dataset(
        dataset_dirs=args.dataset_dirs,
        contrasts=args.contrasts,
        batch_size=args.batch_size,
        normalization_method=args.normalization_method,
        num_workers=args.num_workers,
    )


    # ======================================================
    # 3. INITIALIZE TRAINING
    # ======================================================

    beta_model.initialize_training(
        out_dir=args.out_dir,
        lr=args.lr,
    )


    # ======================================================
    # 4. BEGIN TRAINING
    # ======================================================

    beta_model.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()

import argparse
import sys

from .modules.model import HACA3


def main(args=None):

    args = sys.argv[1:] if args is None else args

    parser = argparse.ArgumentParser(
        description="3D Harmonization with HACA3+."
    )

    # ======================================================
    # DATA
    # ======================================================

    parser.add_argument(
        "--dataset-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Dataset/site directories containing train/val/test folders.",
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
        "--theta-dim",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--eta-dim",
        type=int,
        default=2,
    )


    # ======================================================
    # PRETRAINED MODELS
    # ======================================================

    parser.add_argument(
        "--pretrained-haca3",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--pretrained-eta-encoder",
        type=str,
        default=None,
    )


    # ======================================================
    # TRAINING
    # ======================================================

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    # Full 192 x 224 x 192 volumes currently require
    # batch size 1 on the A40.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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
        f"{text_div} BEGIN 3D HACA3+ TRAINING {text_div}"
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

    haca3 = HACA3(
        beta_dim=args.beta_dim,
        theta_dim=args.theta_dim,
        eta_dim=args.eta_dim,
        pretrained_haca3=args.pretrained_haca3,
        pretrained_eta_encoder=args.pretrained_eta_encoder,
        gpu_id=args.gpu_id,
    )


    # ======================================================
    # 2. LOAD DATASETS
    # ======================================================

    haca3.load_dataset(
        dataset_dirs=args.dataset_dirs,
        contrasts=args.contrasts,
        batch_size=args.batch_size,
        normalization_method=args.normalization_method,
        num_workers=args.num_workers,
    )


    # ======================================================
    # 3. INITIALIZE TRAINING
    # ======================================================

    haca3.initialize_training(
        out_dir=args.out_dir,
        lr=args.lr,
    )


    # ======================================================
    # 4. BEGIN TRAINING
    # ======================================================

    haca3.train(
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

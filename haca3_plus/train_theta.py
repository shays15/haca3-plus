import argparse
import sys

from .modules.theta_model import ThetaModel


def main(args=None):

    args = sys.argv[1:] if args is None else args

    parser = argparse.ArgumentParser(
        description="Pretrain the 3D HACA3+ theta encoder."
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
        "--theta-dim",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for theta contrastive loss.",
    )

    parser.add_argument(
        "--lambda-kld",
        type=float,
        default=1e-4,
        help="Weight applied to theta KLD loss.",
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
        default=4,
        help=(
            "Number of subjects per batch. "
            "Theta contrastive learning benefits from "
            "batch size > 1 because same-contrast images "
            "from different subjects form positive pairs."
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
    # SANITY CHECKS
    # ======================================================

    if args.batch_size < 2:

        print()
        print(
            "[WARNING] Theta contrastive training works best "
            "with batch-size >= 2."
        )

        print(
            "Same-contrast images from different subjects "
            "are used as positive examples."
        )

        print()


    # ======================================================
    # PRINT CONFIGURATION
    # ======================================================

    text_div = "=" * 10

    print(
        f"{text_div} BEGIN THETA ENCODER TRAINING {text_div}"
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
        "Theta dim:",
        args.theta_dim,
    )

    print(
        "Temperature:",
        args.temperature,
    )

    print(
        "KLD weight:",
        args.lambda_kld,
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

    theta_model = ThetaModel(
        theta_dim=args.theta_dim,
        gpu_id=args.gpu_id,
        temperature=args.temperature,
        lambda_kld=args.lambda_kld,
    )


    # ======================================================
    # 2. LOAD DATASETS
    # ======================================================

    theta_model.load_dataset(
        dataset_dirs=args.dataset_dirs,
        contrasts=args.contrasts,
        batch_size=args.batch_size,
        normalization_method=args.normalization_method,
        num_workers=args.num_workers,
    )


    # ======================================================
    # 3. INITIALIZE TRAINING
    # ======================================================

    theta_model.initialize_training(
        out_dir=args.out_dir,
        lr=args.lr,
    )


    # ======================================================
    # 4. BEGIN TRAINING
    # ======================================================

    theta_model.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from aqua_nmnet.evaluation.infer_model import run_inference


def architecture_choice(value):
    try:
        choice = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Architecture choices must be integers, got {value!r}."
        ) from exc

    if not 0 <= choice <= 8:
        raise argparse.ArgumentTypeError(
            f"Architecture choices must be in [0, 8], got {choice}."
        )

    return choice


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run AquaNMNet inference and evaluation for a fixed "
            "13-gene architecture."
        )
    )

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        required=True,
        help="Path to an AquaNMNet checkpoint.",
    )

    parser.add_argument(
        "--arch",
        type=architecture_choice,
        nargs=13,
        required=True,
        metavar="CHOICE",
        help=(
            "Exactly 13 architecture choices. "
            "Each choice must be an integer from 0 to 8."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for enhanced images. "
            "If omitted, the default checkpoint output directory is used."
        ),
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Inference image size. Default: 256.",
    )

    parser.add_argument(
        "--base-channels",
        type=int,
        default=48,
        help="AquaNMNet base channel count. Default: 48.",
    )

    parser.add_argument(
        "--all-images",
        action="store_true",
        help=(
            "Evaluate all images in the configured LSUI input directory "
            "instead of using the test split file."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.image_size <= 0:
        parser.error("--image-size must be greater than zero.")

    if args.base_channels <= 0:
        parser.error("--base-channels must be greater than zero.")

    run_inference(
        ckpt_path=args.checkpoint,
        best_arch=args.arch,
        out_dir=args.output_dir,
        use_test_list=not args.all_images,
        image_size=args.image_size,
        base_channels=args.base_channels,
    )


if __name__ == "__main__":
    main()
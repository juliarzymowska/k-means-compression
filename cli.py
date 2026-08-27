import argparse
from pathlib import Path

from compressor import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Compressing jpg/png images using k-means clustering algorithm.",
    )

    parser.add_argument(
        "-k",
        help="number of clusters to find in an image",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--load",
        help="path to image that should be compressed",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--save",
        help="path where the compressed image should be saved",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-4)

    args = parser.parse_args()

    pipeline(
        load_path=args.load,
        save_path=args.save,
        k=args.k,
        seed=args.seed,
        max_iter=args.max_iter,
        eps=args.eps,
    )
    print(f"Saved the compressed image to: {args.save}")


if __name__ == "__main__":
    main()

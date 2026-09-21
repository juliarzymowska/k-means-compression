import argparse
import sys
from pathlib import Path

from compressor import pipeline
from error import non_negative_int, positive_capped_int, positive_float, positive_int
from kmeans import KMEANS_DEFAULTS


def main():
    parser = argparse.ArgumentParser(
        description="Compressing jpg/png images using k-means clustering algorithm.",
    )

    parser.add_argument(
        "-k",
        help="number of clusters to find in an image",
        type=positive_capped_int,
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
    parser.add_argument(
        "--seed", type=non_negative_int, default=KMEANS_DEFAULTS["seed"]
    )
    parser.add_argument(
        "--max_iter",
        help="max. number of iterations for single run",
        type=positive_int,
        default=KMEANS_DEFAULTS["max_iter"],
    )
    parser.add_argument(
        "--eps",
        help="convergence tolerate",
        type=positive_float,
        default=KMEANS_DEFAULTS["eps"],
    )
    parser.add_argument(
        "-b",
        help="batch size for k-means clustering algorithm",
        type=positive_int,
        default=KMEANS_DEFAULTS["batch_size"],
    )
    parser.add_argument(
        "--backend",
        choices=["scratch", "sklearn"],
        default="scratch",
        help="which k-means implementation to use (default: scratch)",
    )

    args = parser.parse_args()
    try:
        report = pipeline(
            load_path=args.load,
            save_path=args.save,
            k=args.k,
            seed=args.seed,
            max_iter=args.max_iter,
            eps=args.eps,
            batch_size=args.b,
            backend=args.backend,
        )
    except FileNotFoundError:
        print(f"Error: could not find image file: {args.load}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Saved the compressed image to: {args.save}")
    print(
        f"Colors: {report['original_unique_colors']:,} -> {report['compressed_unique_colors']:,}\n"
        f"Size: {report['original_size_bytes']:,} bytes -> {report['compressed_size_bytes']:,} bytes ({report['savings_percent']:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()

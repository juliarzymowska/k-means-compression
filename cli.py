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
    parser.add_argument(
        "-b",
        help="batch size for k-means clustering algorithm",
        default=100_000,
    )
    parser.add_argument(
        "--backend",
        choices=["scratch", "sklearn"],
        default="scratch",
        help="which k-means implementation to use (default: scratch)",
    )

    args = parser.parse_args()

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
    print(f"Saved the compressed image to: {args.save}")
    print(
        f"Colors: {report['original_unique_colors']:,} -> {report['compressed_unique_colors']:,}\n"
        f"Size: {report['original_size_bytes']:,} bytes -> {report['compressed_size_bytes']:,} bytes ({report['savings_percent']:.1f}% smaller)"
    )


if __name__ == "__main__":
    main()

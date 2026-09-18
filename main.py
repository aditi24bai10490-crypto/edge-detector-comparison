import argparse
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from edge_detectors import run_all


def add_gaussian_noise(gray, sigma, rng):
    noise = rng.normal(0, sigma, gray.shape).astype(np.float32)
    noisy = gray.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def edge_density(edge_image):
    return float(np.count_nonzero(edge_image)) / edge_image.size


def save_comparison(results, path, title):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, image) in zip(axes.ravel(), results.items()):
        ax.imshow(image, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Sobel, Canny, LoG and DoG edge detectors."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--noise_levels",
        default="0,5,15,25,40",
        help="Comma-separated Gaussian noise sigma values",
    )
    args = parser.parse_args()

    input_path = Path(args.image)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    gray = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    results = run_all(gray)

    for name, image in results.items():
        filename = f"edges_{name.lower()}.png"
        cv2.imwrite(str(output_dir / filename), image)

    save_comparison(
        results,
        output_dir / "comparison_grid.png",
        "Edge Detector Comparison",
    )

    levels = [float(x.strip()) for x in args.noise_levels.split(",") if x.strip()]
    rng = np.random.default_rng(42)
    rows = []

    clean_results = run_all(gray)

    for sigma in levels:
        noisy = add_gaussian_noise(gray, sigma, rng)
        noisy_results = {}

        for name in clean_results:
            start = time.perf_counter()
            detector_result = run_all(noisy)[name]
            runtime_ms = (time.perf_counter() - start) * 1000
            density = edge_density(detector_result)
            clean_density = edge_density(clean_results[name])
            density_change_pct = (
                abs(density - clean_density) / max(clean_density, 1e-9) * 100
            )

            noisy_results[name] = detector_result
            rows.append(
                [sigma, name, runtime_ms, density, density_change_pct]
            )

        if sigma > 0:
            cv2.imwrite(
                str(output_dir / f"noisy_sigma_{int(sigma)}.png"),
                noisy,
            )

    csv_path = output_dir / "noise_experiment.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("noise_sigma,method,runtime_ms,edge_density,density_change_percent\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")

    # Plot edge-density change as an observable stability measure.
    plt.figure(figsize=(9, 6))
    for name in clean_results:
        subset = [r for r in rows if r[1] == name]
        xs = [r[0] for r in subset]
        ys = [r[4] for r in subset]
        plt.plot(xs, ys, marker="o", label=name)

    plt.xlabel("Gaussian noise sigma")
    plt.ylabel("Absolute change in edge density (%)")
    plt.title("Edge Detector Response to Increasing Noise")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "noise_robustness_plot.png", dpi=150)
    plt.close()

    print("Project completed successfully.")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

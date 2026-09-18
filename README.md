# Edge Detector Comparison & Analysis Tool

A command-line Computer Vision project that compares four classical edge
detectors: Sobel, Canny, Laplacian of Gaussian (LoG), and Difference of
Gaussians (DoG).

## Features

- Manual Sobel convolution using NumPy.
- Canny, LoG and DoG edge detection.
- Side-by-side comparison image.
- Gaussian-noise experiment at configurable noise levels.
- CSV results containing runtime, edge density and edge-density change.
- Fully command-line based; no GUI is required.

## Project structure

```text
edge-detector-comparison/
├── edge_detectors.py
├── main.py
├── requirements.txt
├── sample_images/
├── results/
├── README.md
└── REPORT.md
```

## Requirements

Python 3.8+ is recommended.

## Setup

### Windows

Open Command Prompt in the project folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --image sample_images/synthetic_test.png --output results
```

Custom noise levels:

```bash
python main.py --image sample_images/synthetic_test.png --output results --noise_levels 0,10,20,30,50
```

You can replace the sample image with your own JPG/PNG:

```bash
python main.py --image path/to/image.jpg --output results
```

## Output

The `results/` folder contains:

- `edges_sobel.png`
- `edges_canny.png`
- `edges_log.png`
- `edges_dog.png`
- `comparison_grid.png`
- `noise_experiment.csv`
- `noise_robustness_plot.png`
- noisy test images for non-zero noise levels

## Concepts demonstrated

Convolution, gradient-based edge detection, Gaussian smoothing, Laplacian
operators, Difference of Gaussians, thresholding and image-noise analysis.



## Author

Aditi Soni
B.Tech CSE (AI/ML)
VIT Bhopal University
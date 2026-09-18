import cv2
import numpy as np


def manual_convolution(image, kernel):
    """2-D convolution implemented with NumPy, for grayscale images."""
    image = image.astype(np.float32)
    kernel = np.flipud(np.fliplr(kernel)).astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)

    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded[y:y + kh, x:x + kw]
            output[y, x] = np.sum(region * kernel)
    return output


def sobel_manual(gray):
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gx = manual_convolution(gray, kx)
    gy = manual_convolution(gray, ky)
    magnitude = np.sqrt(gx * gx + gy * gy)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def canny(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    return cv2.Canny(blurred, 50, 150)


def log_detector(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    lap = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
    response = np.abs(lap)
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(response.astype(np.uint8), 40, 255, cv2.THRESH_BINARY)
    return edges


def dog_detector(gray):
    blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
    blur2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
    dog = np.abs(blur1.astype(np.float32) - blur2.astype(np.float32))
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
    _, edges = cv2.threshold(dog.astype(np.uint8), 35, 255, cv2.THRESH_BINARY)
    return edges


def run_all(gray):
    return {
        "Sobel": sobel_manual(gray),
        "Canny": canny(gray),
        "LoG": log_detector(gray),
        "DoG": dog_detector(gray),
    }

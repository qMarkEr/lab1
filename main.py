import numpy as np
import collections
from PIL import Image
from matplotlib import pyplot as plt

def hist(img):
    h, b = np.histogram(img.flatten(), bins = 256, range=(0, 256))
    cdf = h.cumsum()
    cdf = (cdf - cdf.min()) / 255
    return h, b, cdf

def show_grayscale(img, ax, ax1, ax2):
    pixels = img.shape[0] * img.shape[1]
    ax.imshow(img, interpolation='nearest', cmap='gray')

    image_histogram, bins, cdf = hist(img)

    ax1.bar(bins[:-1], image_histogram, width=1.0, color='blue', label='hist')
    ax2.plot(bins[:-1], cdf * pixels, color='red', label='cdf')

def show_rgb(img, ax, ax1, ax2):
    pixels = img.shape[0] * img.shape[1]
    ax.imshow(img, interpolation='nearest')

    image_histogram, bins, cdf = hist(halftone(img))

    ax1.bar(bins[:-1], image_histogram, width=1.0, color='blue', label='hist')
    ax2.plot(bins[:-1], cdf, color='red', label='cdf')

def invert(img):
    return np.invert(img) # same as bitwise not

def halftone(img):
    return np.mean(img, axis=2).astype('uint8')

def noise(img, scale):
    return np.clip(img + np.random.normal(0, scale ** 0.5, img.shape), 0, 255).astype('uint8')

def blur(img, size, intensity):
    radius = size
    size = size * 2 + 1
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * intensity) ** 2) * np.exp(-((x * (1 - size)) ** 2 + (y * (1 - size)) ** 2) / (2 * intensity ** 2)),
        (size, size)
    )
    kernel = kernel / np.sum(kernel)
    height, width = img.shape
    res = np.copy(img)
    np.pad(res, radius, mode='constant', constant_values=0) 
    for i in range(radius, height - radius):
        for j in range(radius, width - radius):
            patch = img[i - radius:i + radius + 1, j - radius:j + radius + 1]
            res[i, j] = np.sum(patch * kernel)
    return res[radius:height - radius, radius: width - radius].astype('uint8')

def equalize(img):
    pixels = img.shape[0] * img.shape[1]
    image_histogram, bins = np.histogram(img.flatten(), bins = 256, range=(0, 256))
    cdf = image_histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (pixels - 1)

    image_equalized = cdf_normalized[img]
    return image_equalized.astype('uint8')

if __name__ == '__main__':
    fig, ax = plt.subplots(2, 3)
    fig, ax1 = plt.subplots(4, 3, sharex=True)

    raw = Image.open('noctrstrgb.jpg')
    raw.load()
    image_data = np.array(raw, dtype='uint8')

    show_rgb(image_data, ax[0][0], ax1[0][0], ax1[1][0])

    inv = invert(image_data)
    show_rgb(inv, ax[0][1], ax1[0][1], ax1[1][1])

    mono = halftone(inv)
    show_grayscale(mono, ax[0][2], ax1[0][2], ax1[1][2])

    noisy = noise(mono, 20)
    show_grayscale(noisy, ax[1][0], ax1[2][0], ax1[3][0])

    blurred = blur(noisy, 10, 40)
    show_grayscale(blurred, ax[1][1], ax1[2][1], ax1[3][1])

    eq = equalize(blurred)
    show_grayscale(eq, ax[1][2], ax1[2][2], ax1[3][2])
    
    plt.show()
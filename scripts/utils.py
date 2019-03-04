import numpy as np
from math import log10

# Convolve over image
def conv(image, kernel):
    # Assuming MxM kernel
    M = len(kernel)
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply edge padding
    # Duplicates the last element in each axis by padding amount
    original_shape = image.shape
    image = np.pad(image, [(int(M/2),int(M/2)) ,(int(M/2),int(M/2))] , 'edge')

    # Convolve filter
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)

    # Operate over entire image
    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            mult = np.multiply(kernel, image[x:x+M, y:y+M])
            output[x,y] = np.sum(mult, axis=(0,1))

    return output

def median(image, M):
    if M < 1:
        return image

    output = np.zeros_like(image)

    # Apply edge padding
    # Duplicates the last element in each axis by padding amount
    original_shape = image.shape
    image = np.pad(image, [(M/2,M/2) ,(M/2,M/2)] , 'edge')

    # Operate over entire image
    for x in range(original_shape[0]):
        for y in range(original_shape[1]):
            output[x,y] = np.median(image[x:x+M, y:y+M])

    return output

def snr(original, noisy):
    original_var = np.var(original)
    noise_var = np.var(noisy)
    return 10 * log10(original_var/noise_var)

def mse(original, new):
    (n, m) = original.shape
    diff = original - new
    error = np.sum(np.square(diff))
    return error / float(m * n)

def circle_mask(img, radius):
    (n, m) = img.shape
    # Find center of image
    center = (m/2, n/2)
    # Generate indices for all coordinates in the image
    idy, idx = np.ogrid[:n, :m]
    # Calculate distances between indices and the image center
    distances = np.sqrt((idx - center[0])**2 + (idy - center[1])**2)
    # Create a mask for indices within the given radius
    mask = distances <= (n / radius)
    # Apply mask to the given image and return
    return np.multiply(mask, img)

def threshold(img, thresh):
    img[img < thresh] = 0
    img[img >= thresh] = 255
    return img

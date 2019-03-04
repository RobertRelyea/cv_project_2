import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import random
import math

random.seed(0)

MAX_ITER = 150

def kmeans(x, k):
    M = x.shape[0]

    # Place initial centers randomly
    centers = []
    for i in range(k):
        center = []
        if len(x.shape) > 1:
            for j in range(x.shape[1]):
                center.append(random.randint(np.min(x), np.max(x)))
            centers.append(center)
        else:
            centers.append(random.randint(np.min(x), np.max(x)))

    prev_centers = []
    new_centers = centers
    assignments = np.zeros(M)
    iteration = 0
    # Perform K-Means
    while((not np.array_equal(new_centers, prev_centers)) and (iteration < MAX_ITER)):
        prev_centers = new_centers
        # Update assignments of each sample
        for sample in range(M):
            # Calculate distance from sample to centers
            distances = []
            for center in prev_centers:
                distances.append(dist(x[sample], center))
            # Assign sample to closest center
            assignments[sample] = np.argmin(distances)

        new_centers = []
        # Calculate cluster means
        for center in range(k):
            # Gather all samples in cluster
            samples = x[np.where(assignments == center)]
            # Calculate average of samples and update cluster center
            if samples.shape[0] > 0:
                new_centers.append(np.average(samples, axis=0))
            else:
                # Keep old center if there are no assigned samples
                new_centers.append(prev_centers[center])
        iteration += 1
        print("Iteration: " + str(iteration) + " Cluster change: " + 
              str(np.sum(np.array(new_centers) - np.array(prev_centers))))
    
    return new_centers, assignments

def dist(x, y):
    distance = 0
    # Check if type is an array or scalar
    if type(x) is np.ndarray:
        for i in range(x.shape[0]):
            distance += (x[i] - y[i])**2
        return math.sqrt(distance)
    else:
        return math.sqrt((x - y)**2)

def kmeans_segment(centers, assignments, shape):
    assignments = assignments.reshape([shape[0], shape[1]])
    assignments = np.uint8(assignments)

    kmeans_image = np.zeros(shape)

    for i in range(assignments.shape[0]):
        for j in range(assignments.shape[1]):
            kmeans_image[i,j] = centers[assignments[i,j]]

    return kmeans_image

image = cv2.imread("../data/Color_images/200.jpg")
b = image[:,:,0]
g = image[:,:,1]
r = image[:,:,2]

# Create 128 bins for each channel
channel_bins = np.arange(0,255,2)
# Flatten channel data
flat_dim = image.shape[0] * image.shape[1]
b_flat = b.reshape(flat_dim)
g_flat = g.reshape(flat_dim)
r_flat = r.reshape(flat_dim)
i_flat = b_flat + g_flat + r_flat / 3
rgb_flat = image.reshape([flat_dim, 3])

# Generate RGB histogram for color image
plt.hist((r_flat,g_flat,b_flat), bins=channel_bins, color=('r','g','b'))
plt.savefig("../figures/part1/histogram.png", transparent=True)

# Perform kmeans on image intensity
centers, assignments = kmeans(i_flat, 3)
i_kmeans = kmeans_segment(centers, assignments, image.shape)

cv2.imwrite("../figures/part1/k3intensity.jpg", np.uint8(i_kmeans))

centers, assignments = kmeans(i_flat, 24)
i_kmeans = kmeans_segment(centers, assignments, image.shape)

cv2.imwrite("../figures/part1/k24intensity.jpg", np.uint8(i_kmeans))

# Perform kmeans on rgb
centers, assignments = kmeans(rgb_flat, 3)
rgb_kmeans = kmeans_segment(centers, assignments, image.shape)

cv2.imwrite("../figures/part1/k3rgb.jpg", np.uint8(rgb_kmeans))

centers, assignments = kmeans(rgb_flat, 24)
rgb_kmeans = kmeans_segment(centers, assignments, image.shape)

cv2.imwrite("../figures/part1/k24rgb.jpg", np.uint8(rgb_kmeans))




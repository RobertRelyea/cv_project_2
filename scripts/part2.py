import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import math
import glob
import re

RUN_DATASET = False
DATASET_PATH = "../data/Color_Images/"

def get_class(filename):
    pattern = re.compile("(\\d+)")
    num = int(pattern.search(filename).group())
    if num < 300:
        return "Architecture"
    elif num < 400:
        return "Bus"
    elif num < 600:
        return "Elephant"
    elif num < 700:
        return "Flower"
    elif num < 800:
        return "Horse"
    elif num < 900:
        return "Mountain"
    else:
        return "Food"

def image_histogram(image):
    # Flatten channel data
    b = image[:,:,0]
    g = image[:,:,1]
    r = image[:,:,2]
    # Generate histograms for each channel
    b_hist, _ = np.histogram(b, bins=32)
    g_hist, _ = np.histogram(g, bins=32)
    r_hist, _ = np.histogram(r, bins=32)

    return np.concatenate((b_hist, g_hist, r_hist), axis=None)

def histogram_intersection(hist1, hist2):
    return np.sum(np.minimum(hist1, hist2))


if RUN_DATASET:
    filenames = glob.glob(DATASET_PATH + "*.jpg")
    histograms = []
    for filename in filenames:
        image = cv2.imread(filename)
        histograms.append(image_histogram(image))
    np.save(DATASET_PATH + "histograms.npy", np.array(histograms))
    np.save(DATASET_PATH + "filenames.npy", np.array(filenames))

histograms = np.load(DATASET_PATH + "histograms.npy")
filenames = np.load(DATASET_PATH + "filenames.npy")

image = cv2.imread(DATASET_PATH + "203.jpg")

matches = np.zeros(histograms.shape[0])
for i in range(histograms.shape[0]):
    im_hist = image_histogram(image)
    matches[i] = histogram_intersection(im_hist, histograms[i]) - np.sum(im_hist)

sorted_idx = np.argsort(np.abs(matches))

for match_idx in range(4):
    print(filenames[sorted_idx[match_idx]])
    print(get_class(filenames[sorted_idx[match_idx]]))
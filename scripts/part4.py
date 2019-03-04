import cv2
import numpy as np
import pdb
from matplotlib import pyplot as plt
import math
import glob
import re
from utils import conv
from scipy import signal

RUN_DATASET = False
DATASET_PATH = "../data/Texture_Images/"

def get_class(filename):
    pattern = re.compile("(?:T)(\d\d)(?:_\d+.jpg)")
    num = int(pattern.search(filename).group(1))
    if num == 1:
        return "Bark"
    elif num == 5:
        return "Wood"
    elif num == 12:
        return "Pebbles"
    elif num == 13:
        return "Wall"
    elif num == 18:
        return "Carpet"
    else:
        return "Plaid"

#                               sigma  theta  lambda gamma
g1  = cv2.getGaborKernel((7,7), 0.3,   0.0,   0.1,   0.5)
g2  = cv2.getGaborKernel((5,5), 0.3,   0.0,   0.1,   0.5)
g3  = cv2.getGaborKernel((3,3), 0.3,   0.0,   0.1,   0.5)
g4  = cv2.getGaborKernel((7,7), 0.3,   0.7,   0.6,   0.5)
g5  = cv2.getGaborKernel((5,5), 0.3,   0.7,   0.6,   0.5)
g6  = cv2.getGaborKernel((3,3), 0.3,   0.7,   0.6,   0.5)
g7  = cv2.getGaborKernel((7,7), 0.3,   0.3,   0.1,   0.5)
g8  = cv2.getGaborKernel((5,5), 0.3,   0.2,   0.9,   0.5)
g9  = cv2.getGaborKernel((3,3), 0.3,   0.9,   0.9,   0.5)
g10 = cv2.getGaborKernel((5,5), 0.3,   0.4,   0.9,   0.5)

plt.subplot(251)
plt.imshow(g1, cmap = 'gray')
plt.subplot(252)
plt.imshow(g2, cmap = 'gray')
plt.subplot(253)
plt.imshow(g3, cmap = 'gray')
plt.subplot(254)
plt.imshow(g4, cmap = 'gray')
plt.subplot(255)
plt.imshow(g5, cmap = 'gray')
plt.subplot(256)
plt.imshow(g6, cmap = 'gray')
plt.subplot(257)
plt.imshow(g7, cmap = 'gray')
plt.subplot(258)
plt.imshow(g8, cmap = 'gray')
plt.subplot(259)
plt.imshow(g9, cmap = 'gray')
plt.subplot(2,5,10)
plt.imshow(g10, cmap = 'gray')
plt.show()

#  .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.    .----------------.  .----------------. 
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |  | .--------------. || .--------------. |
# | |     ______   | || |  ____  ____  | || |      __      | || | ____  _____  | || |    ______    | || |  _________   | |  | | ____    ____ | || |  _________   | |
# | |   .' ___  |  | || | |_   ||   _| | || |     /  \     | || ||_   \|_   _| | || |  .' ___  |   | || | |_   ___  |  | |  | ||_   \  /   _|| || | |_   ___  |  | |
# | |  / .'   \_|  | || |   | |__| |   | || |    / /\ \    | || |  |   \ | |   | || | / .'   \_|   | || |   | |_  \_|  | |  | |  |   \/   |  | || |   | |_  \_|  | |
# | |  | |         | || |   |  __  |   | || |   / ____ \   | || |  | |\ \| |   | || | | |    ____  | || |   |  _|  _   | |  | |  | |\  /| |  | || |   |  _|  _   | |
# | |  \ `.___.'\  | || |  _| |  | |_  | || | _/ /    \ \_ | || | _| |_\   |_  | || | \ `.___]  _| | || |  _| |___/ |  | |  | | _| |_\/_| |_ | || |  _| |___/ |  | |
# | |   `._____.'  | || | |____||____| | || ||____|  |____|| || ||_____|\____| | || |  `._____.'   | || | |_________|  | |  | ||_____||_____|| || | |_________|  | |
# | |              | || |              | || |              | || |              | || |              | || |              | |  | |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |  | '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'    '----------------'  '----------------' 
#  .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.    .----------------.  .----------------. 
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |  | .--------------. || .--------------. |
# | |     ______   | || |  ____  ____  | || |      __      | || | ____  _____  | || |    ______    | || |  _________   | |  | | ____    ____ | || |  _________   | |
# | |   .' ___  |  | || | |_   ||   _| | || |     /  \     | || ||_   \|_   _| | || |  .' ___  |   | || | |_   ___  |  | |  | ||_   \  /   _|| || | |_   ___  |  | |
# | |  / .'   \_|  | || |   | |__| |   | || |    / /\ \    | || |  |   \ | |   | || | / .'   \_|   | || |   | |_  \_|  | |  | |  |   \/   |  | || |   | |_  \_|  | |
# | |  | |         | || |   |  __  |   | || |   / ____ \   | || |  | |\ \| |   | || | | |    ____  | || |   |  _|  _   | |  | |  | |\  /| |  | || |   |  _|  _   | |
# | |  \ `.___.'\  | || |  _| |  | |_  | || | _/ /    \ \_ | || | _| |_\   |_  | || | \ `.___]  _| | || |  _| |___/ |  | |  | | _| |_\/_| |_ | || |  _| |___/ |  | |
# | |   `._____.'  | || | |____||____| | || ||____|  |____|| || ||_____|\____| | || |  `._____.'   | || | |_________|  | |  | ||_____||_____|| || | |_________|  | |
# | |              | || |              | || |              | || |              | || |              | || |              | |  | |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |  | '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'    '----------------'  '----------------' 
#  .----------------.  .----------------.  .----------------.  .-----------------. .----------------.  .----------------.    .----------------.  .----------------. 
# | .--------------. || .--------------. || .--------------. || .--------------. || .--------------. || .--------------. |  | .--------------. || .--------------. |
# | |     ______   | || |  ____  ____  | || |      __      | || | ____  _____  | || |    ______    | || |  _________   | |  | | ____    ____ | || |  _________   | |
# | |   .' ___  |  | || | |_   ||   _| | || |     /  \     | || ||_   \|_   _| | || |  .' ___  |   | || | |_   ___  |  | |  | ||_   \  /   _|| || | |_   ___  |  | |
# | |  / .'   \_|  | || |   | |__| |   | || |    / /\ \    | || |  |   \ | |   | || | / .'   \_|   | || |   | |_  \_|  | |  | |  |   \/   |  | || |   | |_  \_|  | |
# | |  | |         | || |   |  __  |   | || |   / ____ \   | || |  | |\ \| |   | || | | |    ____  | || |   |  _|  _   | |  | |  | |\  /| |  | || |   |  _|  _   | |
# | |  \ `.___.'\  | || |  _| |  | |_  | || | _/ /    \ \_ | || | _| |_\   |_  | || | \ `.___]  _| | || |  _| |___/ |  | |  | | _| |_\/_| |_ | || |  _| |___/ |  | |
# | |   `._____.'  | || | |____||____| | || ||____|  |____|| || ||_____|\____| | || |  `._____.'   | || | |_________|  | |  | ||_____||_____|| || | |_________|  | |
# | |              | || |              | || |              | || |              | || |              | || |              | |  | |              | || |              | |
# | '--------------' || '--------------' || '--------------' || '--------------' || '--------------' || '--------------' |  | '--------------' || '--------------' |
#  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'  '----------------'    '----------------'  '----------------' 


def image_features(image):
    features = []

    features.append(np.average(np.abs(signal.convolve2d(image, g1))))
    features.append(np.average(np.abs(signal.convolve2d(image, g2))))
    features.append(np.average(np.abs(signal.convolve2d(image, g3))))
    features.append(np.average(np.abs(signal.convolve2d(image, g4))))
    features.append(np.average(np.abs(signal.convolve2d(image, g5))))
    features.append(np.average(np.abs(signal.convolve2d(image, g6))))
    features.append(np.average(np.abs(signal.convolve2d(image, g7))))
    features.append(np.average(np.abs(signal.convolve2d(image, g8))))
    features.append(np.average(np.abs(signal.convolve2d(image, g9))))
    features.append(np.average(np.abs(signal.convolve2d(image, g10))))

    return features

def load_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.0
    return image

def chi_distance(feat1, feat2):
    num = np.square(feat1 - feat2)
    denom = feat1 + feat2
    return 0.5 * np.sum(num / denom)

idx = 0
# pdb.set_trace()
if RUN_DATASET:
    filenames = glob.glob(DATASET_PATH + "**/*.jpg", recursive=True)
    features = []
    features_half = []
    features_quarter = []
    for filename in filenames:
        print(str(idx) + " / " + str(len(filenames)))
        image = load_image(filename)
        image_half = cv2.resize(image, None, fx=0.5, fy=0.5)
        image_quarter = cv2.resize(image, None, fx=0.25, fy=0.25)
        features.append(image_features(image))
        features_half.append(image_features(image_half))
        features_quarter.append(image_features(image_quarter))
        idx += 1
    np.save(DATASET_PATH + "gfeatures.npy", np.array(features))
    np.save(DATASET_PATH + "gfeatures_half.npy", np.array(features_half))
    np.save(DATASET_PATH + "gfeatures_quarter.npy", np.array(features_quarter))
    np.save(DATASET_PATH + "filenames.npy", np.array(filenames))

features = np.load(DATASET_PATH + "gfeatures.npy")
features_half = np.load(DATASET_PATH + "gfeatures_half.npy")
features_quarter = np.load(DATASET_PATH + "gfeatures_quarter.npy")
filenames = np.load(DATASET_PATH + "filenames.npy")

image = load_image(DATASET_PATH + "T25_plaid/T25_01.jpg")

matches = np.zeros(features.shape[0])
im_feat = image_features(image)
for i in range(features.shape[0]):
    matches[i] = chi_distance(im_feat, features[i])

sorted_idx = np.argsort(np.abs(matches))

for match_idx in range(4):
    print(filenames[sorted_idx[match_idx]])
    print(get_class(filenames[sorted_idx[match_idx]]))
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

L5 = np.matrix([1, 4, 6, 4, 1])
E5 = np.matrix([-1, -2, 0, 2, 1])
S5 = np.matrix([-1, 0, 2, 0, -1])
R5 = np.matrix([1, -4, 6, -4, 1])

L5E5 = np.transpose(L5) * E5
E5L5 = np.transpose(E5) * L5

L5R5 = np.transpose(L5) * R5
R5L5 = np.transpose(R5) * L5

L5S5 = np.transpose(L5) * S5
S5L5 = np.transpose(S5) * L5

E5R5 = np.transpose(E5) * R5
R5E5 = np.transpose(R5) * E5

E5S5 = np.transpose(E5) * S5
S5E5 = np.transpose(S5) * E5

S5R5 = np.transpose(S5) * R5
R5S5 = np.transpose(R5) * S5

E5E5 = np.transpose(E5) * E5
R5R5 = np.transpose(R5) * R5
S5S5 = np.transpose(S5) * S5


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

    feature = np.abs(signal.convolve2d(image, L5E5)) + np.abs(signal.convolve2d(image, E5L5))
    features.append(np.average(feature))

    feature = np.abs(signal.convolve2d(image, L5R5)) + np.abs(signal.convolve2d(image, R5L5))
    features.append(np.average(feature))

    feature = np.abs(signal.convolve2d(image, L5S5)) + np.abs(signal.convolve2d(image, S5L5))
    features.append(np.average(feature))

    feature = np.abs(signal.convolve2d(image, E5R5)) + np.abs(signal.convolve2d(image, R5E5))
    features.append(np.average(feature))

    feature = np.abs(signal.convolve2d(image, E5S5)) + np.abs(signal.convolve2d(image, S5E5))
    features.append(np.average(feature))

    feature = np.abs(signal.convolve2d(image, S5R5)) + np.abs(signal.convolve2d(image, R5S5))
    features.append(np.average(feature))

    features.append(np.average(np.abs(signal.convolve2d(image, E5E5))))
    features.append(np.average(np.abs(signal.convolve2d(image, R5R5))))
    features.append(np.average(np.abs(signal.convolve2d(image, S5S5))))

    return features

def load_image(filename):
    image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) / 255.0
    return image

def chi_distance(feat1, feat2):
    num = np.square(feat1 - feat2)
    denom = feat1 + feat2
    return 0.5 * np.sum(num / denom)

idx = 0
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
    np.save(DATASET_PATH + "features.npy", np.array(features))
    np.save(DATASET_PATH + "features_half.npy", np.array(features_half))
    np.save(DATASET_PATH + "features_quarter.npy", np.array(features_quarter))
    np.save(DATASET_PATH + "filenames.npy", np.array(filenames))

features = np.load(DATASET_PATH + "features.npy")
features_half = np.load(DATASET_PATH + "features_half.npy")
features_quarter = np.load(DATASET_PATH + "features_quarter.npy")
filenames = np.load(DATASET_PATH + "filenames.npy")

image = load_image(DATASET_PATH + "T25_plaid/T25_01.jpg")
pdb.set_trace()

matches = np.zeros(features.shape[0])
im_feat = image_features(image)
for i in range(features.shape[0]):
    matches[i] = chi_distance(im_feat, features[i])

sorted_idx = np.argsort(np.abs(matches))

for match_idx in range(4):
    print(filenames[sorted_idx[match_idx]])
    print(get_class(filenames[sorted_idx[match_idx]]))
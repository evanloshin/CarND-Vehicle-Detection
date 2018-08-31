from functions_vehicles_only import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import glob
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

################### HYPERPARAMETERS ###################
color_space = 'YUV' # Can be YUV or YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 650] # Min and max in y to search in slide_window()
scale_factor = [0.4, 0.8, 1.2, 1.6] # Scale
heatmap_threshold = 9
################# END HYPERPARAMETERS #################

# Read in cars and not cars
cars = glob.glob('data/vehicles/*/*.png')
notcars = glob.glob('data/non-vehicles/*/*.png')

# Uncomment to reduce the sample size if desired
# to speed up pipeline
cars = shuffle(cars)
notcars = shuffle(notcars)
sample_size = 2000
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]

# extract feature vectors from each category of images
car_features = extract_features(cars, color_conv=color_space,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_conv=color_space,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scalar
X_scaler = StandardScaler().fit(X_train)
# Apply the scalar to X
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))





current_img = mpimg.imread('test_images/test1.jpg')
# Define an list to hold window coordinates
windows = []

# Iterate scales to use search windows of varied size
for scale in scale_factor:
    # Identify vehicles in new image/frame and compile list of detection windows
    windows.extend(find_cars(current_img, color_space, y_start_stop[0], y_start_stop[1], scale,
                           svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))

# create heat map of all detections
heat = np.zeros_like(current_img[:, :, 0]).astype(np.float)
heat = add_heat(heat, windows)
heat = apply_threshold(heat, heatmap_threshold)
heatmap = np.clip(heat, 0, 255)

# draw final boxes based on heat map
draw_img = np.copy(current_img)
labels = label(heatmap)
draw_img = draw_labeled_bboxes(draw_img, labels)

plt.imshow(draw_img)
plt.show()

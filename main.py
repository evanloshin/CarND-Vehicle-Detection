# import python libraries
print('Importing dependencies...')
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import glob
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from moviepy.editor import VideoFileClip
import sys

# import project dependencies
from functions import *
from classes import *


def main(argv):

    ######################## LANE DETECTION HYPERPARAMETERS ########################
    nx_corners = 9  # number of chessboard corners in horizontal direction
    ny_corners = 6  # number of chessboard corners in vertical direction
    x_offset = 400  # horizontal offset for perspective transformation
    y_offset = -10  # vertical offset for perspective transformation
    sobelx_thresh = [20, 255]  # pixel thresholds for horizontal color gradient
    sobely_thresh = [35, 255]  # pixel thresholds for vertical color gradient
    saturation_thresh = [60, 255]  # pixel thresholds for saturation channel in HLS color space
    value_thresh = [40, 255]  # pixel thresholds for value channel in HSV color space
    n_windows = 9  # number of cross-sliding windows
    window_width = 150  # width (pixels) of cross-sliding windows
    min_pix = 40  # minimum number of activated pixels required to recenter window
    curve_margin = 50  # margin around last polynomial fit for including new lane pixels
    smooth_factor = 8  # moving average period for last n polynomial fits
    max_diff = 50  # allowable percent difference between new and previous coefficients
    ###################### END LANE DETECTION HYPERPARAMETERS ######################

    ###################### VEHICLE DETECTION HYPERPARAMETERS #######################
    color_space = 'YUV'  # Can be YUV or YCrCb
    orient = 8  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)  # Spatial binning dimensions
    hist_bins = 32  # Number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off
    y_start_stop = [380, 650]  # Min and max in y to search in slide_window()
    scale_factor = [1.0, 1.4, 1.8]  # scale factor for varying search window sizes
    heatmap_threshold = 13
    window_smoothing = 10 # moving average period for last n frames of windows
    #################### END VEHICLE DETECTION HYPERPARAMETERS #####################



    # take video i/o filenames from terminal
    # try:
    #     input_file = argv[1]
    #     output_file = argv[2]
    # except IndexError:
    #     print("Invalid arguments passed to main.py. Using default i/o filenames.")
    input_file = 'test_videos/project_video.mp4'
    output_file = 'output_videos/output_project_video.mp4'



    ### VEHICLE DETECTION SETUP ###

    print('Training vehicle detection model...')

    # package up hyperparameters for passing to video pipeline
    vehicle_hyperparameters = [color_space, y_start_stop, scale_factor, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, heatmap_threshold]

    # Read in cars and not cars
    cars = glob.glob('data/vehicles/*/*.png')
    notcars = glob.glob('data/non-vehicles/*/*.png')

    # Uncomment to reduce the sample size if desired
    # to speed up pipeline
    cars = shuffle(cars)
    notcars = shuffle(notcars)
    # sample_size = 2000
    # cars = cars[0:sample_size]
    # notcars = notcars[0:sample_size]

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
    svc = SVC(kernel='linear', probability=True)
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Create a moving average object for passing to pipeline
    mvg_avg = movingAverage(window_smoothing)



    ### LANE DETECTION SETUP ###

    # package up hyperparameters for passing to video pipeline
    lane_hyperparameters = [n_windows, window_width, min_pix, curve_margin, smooth_factor, max_diff]

    # grab example driving images and calibration chessboard images
    print('Calibrating camera...')
    Examples = read_image_dir('test_images/')
    Chessboards = read_image_dir('camera_cal/')

    # measure camera distortion
    undist = undistorter(Chessboards, nx_corners, ny_corners)

    # create perspective transformation matrix
    print('Creating detection pipeline...')
    cal = undist.undistort(np.copy(Examples[0]))
    mask = np.int32([[[597, 450], [680, 450], [1020, 675], [290, 675]]])
    txf = transformer(cal, mask, x_offset, y_offset)
    txf.create_transformation()
    #txf.plot_mask() # visual aid for determining coordinates

    # undistort and warp perspective
    undistorted = undist.undistort(cal)
    warped = txf.warp(undistorted)
    #plt.imsave('temp.png', warped)

    # create a binarized image to filter out noise
    thresholds = [sobelx_thresh, sobely_thresh, saturation_thresh, value_thresh]
    binarized = binarize(warped, sobelx_thresh, sobely_thresh, saturation_thresh, value_thresh)

    # pixel to meter conversion
    lanes = laneFinder()
    lanes.generate_unit_conversion(binarized)



    ### VIDEO PIPELINE ###

    # generate video
    output_file = 'output_videos/output_project_video_full.mp4'
    input_file = 'test_videos/project_video_copy.mp4'
    clip = VideoFileClip(input_file)#.subclip(15, 17)
    result = clip.fl_image(lambda frame: video_pipeline(frame, undist, txf, thresholds, lanes, lane_hyperparameters, svc, X_scaler, mvg_avg, vehicle_hyperparameters))
    result.write_videofile(output_file, audio=False)

    # result = video_pipeline(Examples[0], undist, txf, thresholds, lanes, hyperparameters)
    # plt.imsave('test.png', result)

if __name__ == '__main__': main(sys.argv[1:])
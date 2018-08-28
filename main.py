# import python libraries
print('Importing dependencies...')
import numpy as np
import cv2
from matplotlib import pyplot as plt
from moviepy.editor import VideoFileClip
import sys
# import project dependencies
import functions as fn
from classes import undistorter, transformer, laneFinder


def main(argv):

    ######################## HYPERPARAMETERS ########################
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
    ###################### END HYPERPARAMETERS ######################

    # take video i/o filenames from terminal
    try:
        input_file = argv[1]
        output_file = argv[2]
    except IndexError:
        print("Invalid arguments passed to main.py. Using default i/o filenames.")
        input_file = 'test_videos/project_video.mp4'
        output_file = 'output_videos/output_project_video.mp4'

    # package up hyperparameters for passing to video pipeline
    hyperparameters = [n_windows, window_width, min_pix, curve_margin, smooth_factor, max_diff]

    # grab example driving images and calibration chessboard images
    print('Calibrating camera...')
    Examples = fn.read_image_dir('test_images/')
    Chessboards = fn.read_image_dir('camera_cal/')

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
    binarized = fn.binarize(warped, sobelx_thresh, sobely_thresh, saturation_thresh, value_thresh)

    # pixel to meter conversion
    lanes = laneFinder()
    lanes.generate_unit_conversion(binarized)

    # generate video
    output_file = 'output_videos/output_project_video_test.mp4'
    input_file = 'test_videos/project_video.mp4'
    clip = VideoFileClip(input_file).subclip(5, 7)
    result = clip.fl_image(lambda frame: fn.video_pipeline(frame, undist, txf, thresholds, lanes, hyperparameters))
    result.write_videofile(output_file, audio=False)

    # result = fn.video_pipeline(Examples[0], undist, txf, thresholds, lanes, hyperparameters)
    # plt.imsave('test.png', result)

if __name__ == '__main__': main(sys.argv[1:])
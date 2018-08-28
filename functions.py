# import python libraries
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

### grab images from a directory
### returns np array of images
def read_image_dir(path):
    Images = np.array([plt.imread(file) for file in glob.glob(path + '*')])
    return Images

### apply color thresholds to image
### returns binarized image
def binarize(img, sobelx_thresh, sobely_thresh, saturation_thresh, value_thresh):
    # grab S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    # grab v channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    # grab grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # take derivative of the gradient w.r.t. X
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # take derivative of the gradient w.r.t. Y
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # take absolute value of the derivatives
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # rescale back to 255
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    scaled_sobely = np.uint8(255 * abs_sobely / np.max(abs_sobely))
    # translate pixels in each filtered image to binary values
    sobelx_binary = np.zeros_like(scaled_sobelx)
    sobely_binary = np.zeros_like(scaled_sobely)
    sobelx_binary[(scaled_sobelx >= sobelx_thresh[0]) & (scaled_sobelx <= sobelx_thresh[1])] = 1
    sobely_binary[(scaled_sobely >= sobely_thresh[0]) & (scaled_sobely >= sobely_thresh[1])] = 1
    saturation_binary = np.zeros_like(s_channel)
    saturation_binary[(s_channel >= saturation_thresh[0]) & (s_channel <= saturation_thresh[1])] = 1
    value_binary = np.zeros_like(v_channel)
    value_binary[(v_channel >= value_thresh[0]) & (v_channel <= value_thresh[1])] = 1
    # combine binarized images
    combined = np.zeros_like(saturation_binary)
    combined[((sobelx_binary == 1) | (sobely_binary == 1)) | ((saturation_binary == 1) & (value_binary == 1))] = 1
    return combined

### find center of each lane at bottom of frame
### no return
def find_center_points(binary_img):
    # grab bottom half of image
    bottom = binary_img[binary_img.shape[0] // 2:, :]
    # take histogram of activated pixel amplitude
    hist = np.sum(bottom, axis=0)
    # find lane centers
    mid = np.int(hist.shape[0] // 2)
    left_center = np.argmax(hist[:mid])
    right_center = np.argmax(hist[mid:]) + mid
    return [left_center, right_center]

### process frames of video one by one
### returns processed image
def video_pipeline(frame, undist, txf, thresholds, lanes, params):
    undistorted = undist.undistort(frame)
    warped = txf.warp(undistorted)
    binarized = binarize(warped, thresholds[0], thresholds[1], thresholds[2], thresholds[3])
    detected = lanes.find_lanes(binarized, params[0], params[1], params[2], params[3], params[4], params[5])
    result = lanes.project_lines(detected, undistorted, txf)
    # Overlay visualization of lane finding function
    detected_resized = cv2.resize(detected, (0, 0), fx=0.35, fy=0.35)
    y_pos = 20
    x_pos = 20
    result[y_pos: y_pos + detected_resized.shape[0], x_pos: x_pos + detected_resized.shape[1]] = detected_resized
    # Overlay radius measurements
    radius = lanes.get_radius()
    text = 'Radius of curvature: ' + str(int(radius)) + 'm'
    cv2.putText(result, text, (550, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), thickness=3)
    # Overlay center distance
    distance = lanes.get_center_distance(warped)
    text2 = 'Distance from center: ' + str(round(distance, 1)) + 'm'
    cv2.putText(result, text2, (550, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), thickness=3)
    return result
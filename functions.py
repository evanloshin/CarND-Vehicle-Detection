# import python libraries
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label

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
def video_pipeline(frame, undist, txf, thresholds, lanes, lane_params, svc, X_scaler, window_smoothing, vehicle_params):

    ## lane detection
    lane_img = np.copy(frame)
    undistorted = undist.undistort(lane_img)
    warped = txf.warp(undistorted)
    binarized = binarize(warped, thresholds[0], thresholds[1], thresholds[2], thresholds[3])
    detected = lanes.find_lanes(binarized, lane_params[0], lane_params[1], lane_params[2], lane_params[3], lane_params[4], lane_params[5])
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

    ## vehicle detection
    color_space = vehicle_params[0]
    y_start_stop = vehicle_params[1]
    scale_factor = vehicle_params[2]
    orient = vehicle_params[3]
    pix_per_cell = vehicle_params[4]
    cell_per_block = vehicle_params[5]
    spatial_size = vehicle_params[6]
    hist_bins = vehicle_params[7]
    heatmap_threshold = vehicle_params[8]

    vehicle_img = np.copy(undistorted)
    # Define an list to hold window coordinates
    windows = []
    # Iterate scales to use search windows of varied size
    for scale in scale_factor:
        # Identify vehicles in new image/frame and compile list of detection windows
        windows.extend(find_cars(vehicle_img, color_space, y_start_stop[0], y_start_stop[1], scale,
                                 svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins))
    recent_windows = sum(window_smoothing.next(windows), [])
    # create heat map of all detections
    heat = np.zeros_like(vehicle_img[:, :, 0]).astype(np.float)
    heat = add_heat(heat, recent_windows)
    heat = apply_threshold(heat, heatmap_threshold)
    heatmap = np.clip(heat, 0, 255)
    # draw final boxes based on heat map
    labels = label(heatmap)
    result = draw_labeled_bboxes(result, labels)

    return result

# convert an image to new cv2 color space
# defined by parameter "conv"
def convert_color(img, conv):
    conv_str = 'cv2.cvtColor(img, cv2.COLOR_RGB2' + conv + ')'
    return eval(conv_str)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys',
                                  transform_sqrt=False,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=False,
                       visualize=vis, feature_vector=feature_vec)
        return features


# downsize a passed image and return a flat list of
# pixel values from each color channel
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


# take frequency histograms of the pixel values in each channel
# of the passed image and return a single vector
def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Define a single function that can extract features
# using hog sub-sampling and make predictions
def find_cars(img, color_conv, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    draw_img = np.copy(img)
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, color_conv)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # Define list of window coordinates
    box_list = []

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                box_list.append(((xbox_left, ytop_draw + ystart),
                                 (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

    return box_list


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_conv, spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion
        feature_image = convert_color(image, color_conv)
        if spatial_feat:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
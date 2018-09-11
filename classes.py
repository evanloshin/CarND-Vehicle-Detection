# import python libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import deque
# import project dependencies
from functions import *

class movingAverage():

    ## constructor
    def __init__(self, nb_periods):
        self.queue = deque(maxlen=nb_periods)

    def next(self, val):
        self.queue.append(val)
        return list(self.queue)

class undistorter():

    ## constructor
    def __init__(self, Images, nx_corners, ny_corners):
        # attributes
        self.mtx = None
        self.dist = None
        # Init arrays for storing 3d object and 2d image points
        ObjPts = []
        ImgPts = []
        # prepare object points (adapted these two nasty lines from Udacity)
        objp = np.zeros((ny_corners * nx_corners, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx_corners, 0:ny_corners].T.reshape(-1, 2)
        # Step through chessboard images
        for img in Images:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx_corners, ny_corners), None)
            # Check if corners were found
            if ret == True:
                # Draw the corners
                cv2.drawChessboardCorners(img, (nx_corners, ny_corners), corners, ret)
                # Append object and image points
                ObjPts.append(objp)
                ImgPts.append(corners)
        # grab calibration image dimensions
        imgShape = (Images[0].shape[1], Images[0].shape[0])
        # calculate calibration parameters
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(ObjPts, ImgPts, imgShape, None, None)

    ## undistort an image
    ## returns undistorted image
    def undistort(self, img):
        result = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return result


class transformer():

    ## constructor
    def __init__(self, img, pts, x_offset, y_offset):
        self.example = img
        self.mask = pts
        self.x = x_offset
        self.y = y_offset
        self.M = None
        self.Minv = None


    ### display polygon region for perspective transform
    ### no return
    def plot_mask(self):
        cv2.polylines(self.example, self.mask, True, (255, 0, 0), thickness=4)
        fig = plt.imshow(self.example)
        plt.grid()
        plt.show()

    ### calculate transformation matrix
    ### no return
    def create_transformation(self):
        # get image dims
        img_size = self.example.shape
        # Cast source points to float
        src = np.float32(self.mask)
        # Define 4 destination points
        dst = np.float32([[self.x, self.y], [img_size[1] - self.x, self.y],
                          [img_size[1] - self.x, img_size[0] - self.y],
                          [self.x, img_size[0] - self.y]])
        # Get M, the transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        # Get Minv, the inverse matrix for unwarping later on
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    ### transform image to top-down perspective
    ### returns warped image
    def warp(self, img):
        result = cv2.warpPerspective(img, self.M, (img.shape[1], img.shape[0]))
        return result

    ### transform image back to original perspective
    ### returns unwarped image
    def unwarp(self, img):
        result = cv2.warpPerspective(img, self.Minv, (self.example.shape[1], self.example.shape[0]))
        return result


class laneFinder():

    ### constructor
    def __init__(self, mx_per_pix=None, my_per_pix=None):

        # conversions from pixels to meters
        self.mx = mx_per_pix
        self.my = my_per_pix
        # most recent center points of each line
        self.best_fit = None
        # left and right lane objects
        self.left_lane = lane()
        self.right_lane = lane()
        # left and right lane pixels
        self.x_left = None
        self.y_left = None
        self.x_right = None
        self.y_right = None
        # middle curve
        self.middle_x = None
        # moving average for radius measurement
        self.recent_r = movingAverage(10)


    ### generate pixel to meter conversions based on a sample image
    ### no return
    def generate_unit_conversion(self, binary_img, lane_width_meters=3.7, lane_distance_meters=10):
        # grab lane centers
        lane_centers = find_center_points(binary_img)
        # calculate lane width in pixels
        pixel_width = lane_centers[1] - lane_centers[0]
        # calculate conversion factors
        self.mx = lane_width_meters / pixel_width
        self.my = lane_distance_meters / binary_img.shape[0]


    ### search for lane pixels using x-sliding window technique
    ### returns image of technical lane depictions
    def find_lanes(self, binary_image, n_windows, window_width, min_pix_per_window, curve_margin, smooth_factor, max_diff):

        binary_img = np.copy(binary_image)
        # grab lane centers
        lane_centers = find_center_points(binary_img)
        # use centers as starting point for windows
        left_window_base = lane_centers[0]
        right_window_base = lane_centers[1]
        # set window height based on image and number of windows
        window_height = np.int(binary_img.shape[0]//n_windows)
        # identify x and y coordinates of all activate pixels
        nonzero = binary_img.nonzero()
        nonzero_x = np.array(nonzero[1])
        nonzero_y = np.array(nonzero[0])
        # initialize lists of lane pixel coordinates
        left_lane_pix = []
        right_lane_pix = []
        # create image to draw on
        result_img = np.zeros_like(binary_img)
        result_img = np.dstack((result_img, result_img, result_img))
        # iterate through windows vertically
        for window_idx in range(n_windows):
            # calculate window y position
            y_low = binary_img.shape[0] - (window_idx + 1) * window_height
            y_high = binary_img.shape[0] - window_idx * window_height
            # keep windows within search margin of last polynomial fit
            left_window_base = self.left_lane.verify_window(left_window_base, y_high, curve_margin, smooth_factor)
            right_window_base = self.right_lane.verify_window(right_window_base, y_high, curve_margin, smooth_factor)
            # calculate window x positions
            x_left_line_low = int(left_window_base - window_width / 2)
            x_left_line_high = int(left_window_base + window_width / 2)
            x_right_line_low = int(right_window_base - window_width / 2)
            x_right_line_high = int(right_window_base + window_width / 2)
            # draw windows
            cv2.rectangle(result_img, (x_left_line_low, y_high), (x_left_line_high, y_low), (230,230,0), 3)
            cv2.rectangle(result_img, (x_right_line_low, y_high), (x_right_line_high, y_low), (230, 230, 0), 3)
            # record activated pixels within window
            left_window_pix = ((nonzero_y >= y_low) & (nonzero_y < y_high) & (nonzero_x >= x_left_line_low) & (
                    nonzero_x < x_left_line_high)).nonzero()[0]
            right_window_pix = ((nonzero_y >= y_low) & (nonzero_y < y_high) & (nonzero_x >= x_right_line_low) & (
                    nonzero_x < x_right_line_high)).nonzero()[0]
            left_lane_pix.append(left_window_pix)
            right_lane_pix.append(right_window_pix)
            # recenter window if min pixel threshold is met
            if len(left_window_pix) >= min_pix_per_window:
                left_window_base = np.int(np.mean(nonzero_x[left_window_pix]))
            if len(right_window_pix) >= min_pix_per_window:
                right_window_base = np.int(np.mean(nonzero_x[right_window_pix]))
        # combine list of lists
        left_lane_pix = np.concatenate(left_lane_pix)
        right_lane_pix = np.concatenate(right_lane_pix)
        # extract x and y values
        self.x_left = nonzero_x[left_lane_pix]
        self.y_left = nonzero_y[left_lane_pix]
        self.x_right = nonzero_x[right_lane_pix]
        self.y_right = nonzero_y[right_lane_pix]
        # highlight lane pixels
        result_img[self.y_left, self.x_left] = [200, 50, 50]
        result_img[self.y_right, self.x_right] = [50, 50, 255]
        # fit curves to each set of pixels
        result_img = self.left_lane.fit_curve(result_img, self.x_left, self.y_left, smooth_factor, max_diff)
        result_img = self.right_lane.fit_curve(result_img, self.x_right, self.y_right, smooth_factor, max_diff)
        # save middle x values
        self.middle_x = np.array(self.left_lane.fit_x) + (np.array(self.right_lane.fit_x) - np.array(self.left_lane.fit_x)) / 2
        self.middle_x = self.middle_x.astype(np.int32)
        # format image
        result_img[np.where((result_img == [0, 0, 0]).all(axis=2))] = [240, 240, 240]
        return result_img


    ### calculate radius of lane curvature for most recent fit
    ### return radius in meters
    def get_radius(self):
        coeff = np.average((self.left_lane.last_coeff, self.right_lane.last_coeff), axis=0)
        y_eval = np.max(self.left_lane.plot_y)
        r = ((1 + (2 * coeff[0] * y_eval * self.my + coeff[1]) ** 2) ** 1.5) / abs(
            2 * coeff[0])
        return np.average(self.recent_r.next(r))



    ### project lines back down onto original image
    ### returns unwarped image with lane indicators
    def project_lines(self, detected_img, original_img, transformer_obj):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(detected_img[:, :, 0]).astype(np.uint8)
        warped_lanes = np.dstack((warp_zero, warp_zero, warp_zero))
        left_pts = self.left_lane.line_pts
        right_x = self.right_lane.fit_x
        right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, self.right_lane.plot_y])))])
        final_pts = np.hstack((left_pts, right_pts))
        middle_pts = np.array([np.transpose(np.vstack([self.middle_x, self.left_lane.plot_y]))])
        # Draw the lane area onto the warped blank image
        cv2.fillPoly(warped_lanes, np.int_([final_pts]), (40, 200, 40))
        # Draw center line onto image
        cv2.polylines(warped_lanes, np.int_([middle_pts]), False, (0,255,0), thickness=5)
        # Highlight the detected lane pixels
        warped_lanes[self.y_left, self.x_left] = [255, 0, 0]
        warped_lanes[self.y_right, self.x_right] = [0, 0, 255]
        # Warp the blank back to original image space
        unwarped = transformer_obj.unwarp(warped_lanes)
        # Combine the result with the original image
        result = cv2.addWeighted(original_img, 1, unwarped, 0.5, 0)
        return result


    ### calculate distance of vehicle from center
    ### returns distance in meters
    def get_center_distance(self, warped):
        veh_center = warped.shape[1] / 2
        pixels = self.middle_x[0] - veh_center
        meters = pixels * self.mx
        return meters



class lane():

    ### constructor
    def __init__(self):

        # store several frames of pixels found in cross-sliding windows
        self.historical_x = []
        self.historical_y = []
        # store last polynomial coefficients
        self.last_coeff = None
        # line space for plotting
        self.plot_y = None
        # curve values
        self.line_pts = None
        self.fit_x = None
        # recent radius measurements
        self.R = []
        # moving average period
        self.period = None


    ### fit polynomial to lane pixels
    ### return image with curve drawn
    def fit_curve(self, img, X, Y, period, max_diff):
        # store period
        self.period = period
        # store lane pixels
        self.historical_x.insert(0, (X))
        self.historical_y.insert(0, (Y))
        # update moving average of pixels
        if len(self.historical_x) > period:
            del(self.historical_x[-1])
            del(self.historical_y[-1])
        # concatenate all the pixels over the last n frames
        all_x = np.copy(self.historical_x)
        all_y = np.copy(self.historical_y)
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        # generate line space for plotting
        self.plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
        # fit curve
        new_coeff = np.polyfit(all_y, all_x, 2)
        # only accept coefficients within percent of previous
        #if len(self.historical_x) < period:
        self.last_coeff = new_coeff
        # else:
        #     diff = np.sum(np.abs(np.array(self.last_coeff) - np.array(new_coeff)) / abs(self.last_coeff)) / 3
        #     if diff <= max_diff / 100:
        #         self.last_coeff = new_coeff

        # draw curve
        self.fit_x = self.last_coeff[0] * self.plot_y**2 + self.last_coeff[1] * self.plot_y + self.last_coeff[2]
        self.line_pts = np.array([np.transpose(np.vstack([self.fit_x, self.plot_y]))])
        cv2.polylines(img, np.int_([self.line_pts]), False, (0, 255, 0), thickness=4)
        return img


    ### make sure x-sliding window stays near last fit
    ### returns x coordinate of window base
    def verify_window(self, window_x, window_y, max_margin, period):
        if len(self.historical_x) == period:
            expected_x = self.last_coeff[0] * window_y**2 + self.last_coeff[1] * window_y + self.last_coeff[2]
            if abs(expected_x - window_x) > max_margin:
                return int(expected_x)
            else:
                return window_x
        else:
            return window_x

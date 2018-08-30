# Vehicle Detection
### This project identifies lanes and vehicles in driving video taken from a single front-facing camera. This is the final project in term one of Udacity's Self-Driving Car Nanodegree.

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Train a Linear SVM classifier with the full feature vectors to detect vehicles.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a [video stream](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

## Grading Rubric
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function *?????* of [functions.py](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/functions.py).

First I read in all the *vehicle* and *non-vehicle* training images. Here's some examples of what they look like. 

![alt text][image1]

I then explored how different color spaces and parameters affect the extraction of HOG features using `skimage.hog()`. I grabbed random images from each of the two classes and displayed them to get a feel for what the HOG output looks like. While exploring colorspaces, I held constant the HOG parameters `orientations`, `pixels_per_cell`, and `cells_per_block` and visa versa.

Here is an example using the `YUV` color space and HOG parameters `orientations=8`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, and `spatial_size=(32, 32)`:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

With the SVM implemented, I tried several combinations of colorspaces and parameters to tune the model. Here are some of the intermediate paramters and their outcomes.

XXXXXXXXXX
XXXXXXXXXX
XXXXXXXXXX

The best parameters I retained for creating feature vectors are XXXXXX.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implement a sliding window search in function `find_cars()` of [functions.py](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/functions.py).

Search parameters include `y_start_stop`, `, XXXXX. Here are several intermediary results from tuning the search function:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, my pipeline...

A few opportunities exist to further improve my implementation of the project.

For starters, addressing overfitting would benefit the project. The SVC accuracy is very high (>99%) even on a smaller samples of the data. Udacity points out the supplied training data includes time-series images, so accuracy is being calculated on test images nearly identical to those in the training set. At present, this pipeline does not implement a solution to deal with the time-series data.

Also, the final video detects the black colored vehicle more reliably than the white one. Given a solution to overfitting, a fresh examination of which color channels to include in the feature vector might better amplify white cars. If that's not sufficient, another fix is gathering additional labeled images of white cars for training the SVC model.

Last, poor code performance could severely impact this project's applicability to actual self-driving cars. My pipeline averages 0.06 frames per second, mostly attributed to sklearn's HOG function, but optimizing my code could bring some improvement. While industry sampling rates are not publicly known, any rate less than 1 frame per second is clearly insufficient. Granted the compute in today's autonomous vehicles exceeds my circa 2010 MacBook, they must handle many more tasks besides those implemented here in real-time. A neural network approach would more substantially reduce processing time by eliminating HOG gradients entirely, but not without the tradeoff of replacing a clearly interpretable approach with a somewhat ambigous black box.


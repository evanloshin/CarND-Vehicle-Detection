# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### This project identifies lanes and vehicles in driving video taken from a single front-facing camera. This is the final project in term one of Udacity's Self-Driving Car Nanodegree.

---

[Final video submission](output_videos/output_project_video_full.mp4) <-- click here

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images.
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Train a Linear SVM classifier with the full feature vectors to detect vehicles.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a [video stream](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/test_videos/project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_and_noncar.png
[image2]: ./output_images/HOG_example.png
[image3]: ./output_images/window_scales.png
[image4]: ./output_images/svc_thresh.png
[image5]: ./output_images/heatmaps.png
[image6]: ./output_images/labeling.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

## Grading Rubric
##### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.

### Histogram of Oriented Gradients (HOG)

##### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function `get_hog_features()` of [functions.py](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/functions.py).

First I read in all the *vehicle* and *non-vehicle* training images. Here's some examples of what they look like.

![alt text][image1]

I then explored how different color spaces and parameters affect the extraction of HOG features using `skimage.hog()`. I grabbed random images from each of the two classes and displayed them to get a feel for what the HOG output looks like. While exploring colorspaces, I held constant the HOG parameters `orientations`, `pixels_per_cell`, and `cells_per_block` and visa versa.

Below's an example using the `YUV` color space and HOG parameters `orientations=8`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`, and `spatial_size=(16, 16)`. I explain how I chose these parameters in the next section.

![alt text][image2]

##### 2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I use the function `extract_features()` to combine the results of the HOG transform, color features, and histogram frequencies into a single feature vector for each image. Then, I apply a scalar transform to normalize the vectors before feeding them to a linear SVM classifier. I train the SVM in lines `119-125` of [main.py](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/main.py) to label feature vectors as belonging to either the car or not-car class.

With the SVM implemented, I tried several combinations of colorspaces and parameters to tune the model. My objective is to maximize the classifier's accuracy on 20% of reserved labeled data. Here are some of the intermediate parameters and their outcomes:
   
| Trial | Colorspace | Orientations | Pixels/cell | Cells/Block | Accuracy |
|:-----:|:----------:|:------------:|:-----------:|:-----------:|:--------:|
|   #1  |     RGB    |       8      |    (8,8)    |   (32,32)   |  98.25%  |
|   #2  |     HLS    |       8      |    (8,8)    |   (32,32)   |  98.38%  |
|   #3  |     YUV    |       8      |    (8,8)    |   (32,32)   |  98.75%  |
|   #4  |     YUV    |       8      |    (8,8)    |   (16,16)   |  98.88%  |
**Ran with a 2,000 image subset of the data for comparison. The final result uses the whole dataset for better accuracy.*

### Sliding Window Search

##### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I implement a sliding window search in function `find_cars()` of [functions.py](https://github.com/evanloshin/CarND-Vehicle-Detection/blob/master/functions.py).

Search parameters are `y_start_stop = (380, 650)` and `cells_per_step = 4`. I use the first parameter to reduce the search space to only a bottom portion of the image about as distant as the lane tracking pipeline to increase speed and filter false detections. The second parameter, four pixels per step, was simply as low as I was willing to go in terms of my appetite for slow performance.

My pipeline uses three different window sizes to increase performance throughout the depth of perspective. Here are what the window search looks like for each of the three sizes:

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

One way I filter out false positives is by thresholding the SVC prediction confidence. Rather than using sklearn's `LinearSVC()` class, I actually opt for the standard `SVC()` class but pass the parameters `kernel='linear'` and `probability=True`. This way, I can call the method `SVC.predict_proba()` to check the confidence is above a certain threshold. This technique yielded the following results:

![alt text][image4]

---

### Video Implementation

#### Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In the first pass at running the video, I noticed a lot of false positive detections as well as deficiency identifying the white car. To help with this, I implemented the `moving_average` class to store the last 10 frames worth of windows in conjunction with increasing the threshold for minimum number of positive detections.

From the positive detections, I created a heatmap in the function `add_heat()` and then thresholded that map in `apply_threshold()` to identify vehicle positions while reducing false detections. This is redundant with the SVC filter, however I found the combination of both techninques was more effective in the final video than either one alone. The example below shows how thresholding, by itself, reduces false detetcions in my pipeline:

![alt text][image5]
  
Using `scipy.ndimage.measurements.label()` in my `label()` function helps identify individual blobs in the heatmap.  I finally construct bounding boxes around each blob in the function `draw_labeled_bboxes()`. Here's an example of the final bounding boxes:

![alt text][image6]

---

### Discussion

##### Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Overall, my pipeline tracks the two prominent vehicles consistently throughout the video. Opportunities for improvement include more accuractely bounding the white vehicle as well as speeding up code performance.

For starters, addressing overfitting would benefit the project. The SVC accuracy is very high (>99%) even on a smaller samples of the data. Udacity points out the supplied training data includes time-series images, so accuracy is being calculated on test images nearly identical to those in the training set. At present, this pipeline does not implement a solution to deal with the time-series data.

Also, the final video detects the black colored vehicle more reliably than the white one. Given a solution to overfitting, a fresh examination of which color channels to include in the feature vector might better amplify white cars. If that's not sufficient, another fix is gathering additional labeled images of white cars for training the SVC model.

Last, poor code performance could severely impact this project's applicability to actual self-driving cars. My pipeline averages 0.06 frames per second, mostly attributed to sklearn's HOG function, but optimizing my code could bring some improvement. While industry sampling rates are not publicly known, any rate less than 1 frame per second is clearly insufficient. Granted the compute in today's autonomous vehicles exceeds my circa 2010 MacBook, they handle many other tasks. A neural network approach would reduce processing time by eliminating HOG gradients entirely, but not without the tradeoff of replacing a clearly interpretable approach with a somewhat ambigous black box.

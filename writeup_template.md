## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./undistorted_camera_cal/calibration1.jpg "Undistorted"

[image2]: ./undistorted_test_images/straight_lines1.jpg "Road Transformed"

[imgrgb]: ./output_images/rgb_colourspace.png "RGB Example"

[imghls]: ./output_images/hls_colourspace.png "HLS Example"

[image4]: ./output_images/bird_eye_view.jpg "Warp Example"

[image5]: ./output_images/hist_lower_half.jpg "Fit Visual"

[image5_1]: ./output_images/sliding_window.jpg "Sliding Window"

[image5_2]: ./output_images/straight_lines_drawn.jpg "Lines Drawn"

[formula]: ./output_images/formula.png "formula img"

[invwrap]: ./output_images/final.jpg "bound drawn"

[image6]: ./output_images/final_result.png "Output"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "
./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world.
Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each
calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy
of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (
x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using
the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()`
function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 95 through
99 in `utils.py`). Here's an example of my output for this step. I compared RGB color channels and HLS color channels to
get a good visual of effects of different color channels

![RGB][imgrgb]
![HLS][imghls]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `transform_n_warp()`, which appears in lines 1 through
8 in the file `utils.py` (./utils.py) and also a couple of opencv functions called `getPerspectiveTransform()`
and `warpPerspective()`. The `transform_n_warp()` function takes as inputs an image (`img`), as well as matrix M which
is warping matrix. I chose the hardcode the source and destination points in the following manner:

```python
src_points = np.float32([[0.2 * col, 0.9 * row],
                         [0.45 * col, 0.64 * row],
                         [0.55 * col, 0.64 * row],
                         [0.8 * col, 0.9 * row]])
dst_points = np.float32([[0.2 * col, 0.9 * row],
                         [0.2 * col, 0 * row],
                         [0.8 * col, 0 * row],
                         [0.8 * col, 0.9 * row]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 270, 674      |  270, 674     |
| 579, 460      |   270, 0      |
| 720, 460      |   1035, 0     |
| 1035, 674     |  1035, 674    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image
and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Histogram (line 156 of project.py) was used to detect the lane regions in the perspective transformed image as shown in
the below image:
![alt text][image5]

Next step was the sliding window part. From lines 177 to 261, with comments contains the algorithm. The image below
shows sliding window and polyfit result

![slidingwindow][image5_1]
![strlines][image5_2]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 119 through 126 in my code in `utils.py` with formula below

![formula][formula]

After this formula is used, lane boundaries are drawn and inverse wrapped the image:

![invwrap][invwrap]


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 388 through 408 in my code in `project.py` in the function `display_on_frame()`. Here is
an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_solved.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and
how I might improve it if I were going to pursue this project further.  

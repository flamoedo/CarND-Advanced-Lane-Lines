## Writeup

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

[image1]: ./output_images/undistorted.png "Undistorted"
[image1_1]: ./output_images/undistorted_camera.png "Undistorted Camera"
[image2]: ./output_images/thresholds.png "Thresholds"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/lines_detect.png "Lines Detect"
[image6]: ./output_images/result_image.png "Output"
[video1]: ./output_images/project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Here I'll briefly state how was computed the camera matrix and distortion coefficients and provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Finding Lane Lines.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. This is an example of a distortion-corrected image.

![alt text][image1_1]

#### 2. Now I used color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 8 and 9 of the notebook).  Here's an example of my output for this step.  

![alt text][image2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in cell 11 of the notebook.  The `warp()` function takes as inputs an image (`img`) that must be already undistorted by the function `cal_undistort()`.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
            [[670, 442],
            [1108, 720],
            [205, 720],
            [607, 442]])
        
dst = np.float32(
             [[900, 0],
             [900, 700],
             [350, 700],
             [350, 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 670, 442      | 900, 0    | 
| 1108, 720     | 900, 700  |
| 205, 720      | 350, 700  |
| 607, 442      | 350, 0    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Identified lane-line pixels and fit their positions with a polynomial.

Then in cell 16 of notebook, the function `detect_lanes()` uses a polinomial to draw the lane lines detect on the image.
There are also a fuction called `detect_lanes_next_frame()`, that is used after the first detection, to improve the performance, restricting the search area.
When the lines become not parallel, the first detection is applied again. This process can be seen in the line 378 of the file `detect_lanes.py`

![alt text][image5]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the cell 23 and 25 of the notebook.

#### 6. Result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the cell 27 of the notebook.  
To process the video, I implemented a function called `pipeline()` in the line 348 of the file `detect_lanes.py`
Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

Here's a [link to my video result](./output_images/project_video_output.mp4)
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/2XtvhMY9kGs/0.jpg)](http://www.youtube.com/watch?v=2XtvhMY9kGs)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There was some problems with detection, and I had to experiment adjusts in many images, and finally I end up creating a start stop function, so I could adjst the parameters, in the problematic areas of the video using cursors (line 274 of `detect_lanes.py`).

There was also to many interferences on detection, I realise this by overllapping the detection image on the original image (line 234 of `detect_lanes.py`), so I had to create masks to clear some parts of the image (line 366 of `detect_lanes.py`).



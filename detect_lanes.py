import numpy as np
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip

# Function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image


def cal_undistort(img):
    # Use cv2.calibrateCamera() and cv2.undistort()
    # undist = np.copy(img)  # Delete this line
    
    img_size = (img.shape[1], img.shape[0])
    
    global loaded
    
    global mtx, dist, rvecs, tvecs
    
    if(not loaded):
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
    
        loaded, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


def binary_image(img, s_thresh=(150, 255), sx_thresh=(20, 100)):

    # Convert to HLS color space and separate the S channel
    # Note: img is the undistorted image
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    
    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what might work better
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = sx_thresh[0]
    thresh_max = sx_thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    # Threshold color channel
    s_thresh_min = s_thresh[0]
    s_thresh_max = s_thresh[1]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary, ))
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    
    return s_binary, sxbinary, combined_binary


def warp(img):
        
    M = cv2.getPerspectiveTransform(src, dst)
    
    img_size = (img.shape[1], img.shape[0])
    
    return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

import matplotlib.pyplot as plt


def detect_lanes(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return leftx, lefty, rightx, righty, left_fit, right_fit, left_lane_inds, right_lane_inds, left_fitx, right_fitx, ploty

#Skip the sliding windows step once you know where the lines are
#Now you know where the lines are you have a fit! In the next frame of video you don't need to do a blind search again, but instead you can just search in a margin around the previous line position like this:

def detect_lanes_next_frame(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, left_fitx, right_fitx, left_fit, right_fit, ploty


def draw_lines(warp, undist, left_fitx, right_fitx, ploty, left_curverad, right_curverad, car_pos):
    
    
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warp).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Draw left line
    cv2.polylines(color_warp, np.int_(pts_left), False, (0,255,255), 3) 
    
    # Draw right line
    cv2.polylines(color_warp, np.int_(pts_right), False, (255,255,0), 3) 
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (warp.shape[1], warp.shape[0]))
    
    
    if (left_curverad > 1500 or right_curverad > 1500):
        radius_text = 'Straight road, Pos:{:3.2f}m'.format(car_pos)
    else:        
        radius_text = 'Left:{:5.0f}m, Right:{:5.0f}m, Pos:{:3.2f}m'.format(left_curverad, right_curverad, car_pos)
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(newwarp, radius_text, (10,100), font, 1,(255,255,255),2)
    
    # Draw binary images
    unwarp = cv2.warpPerspective(np.dstack((warp_zero, warp_zero, warp)) * 255, Minv, (warp.shape[1], warp.shape[0]))
    
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.5, 0)
    
    result = cv2.addWeighted(result, 1, unwarp, 0.8, 0)
    
    return result

src = np.float32(
            [[670, 442],
            [1108, 720],
            [205, 720],
            [607, 442]])
        
dst = np.float32([[900, 0],
             [900, 700],
             [350, 700],
             [350, 0]])


def measure_angle(leftx, lefty, rightx, righty, ploty):

    y_eval = np.max(ploty)
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad


def car_position(right, left, warp):
    return (right - left - warp/2) * 3.7/700    

def nothing(x):
    pass

def run_video(video_path):
        
    cap = cv2.VideoCapture(video_path)
    
    ret = True
    
    global loaded, left_fit, right_fit, first_frame
    
    first_frame = True    

    loaded = False
    
    global s_thresh_min, s_thresh_max, sx_thresh_min, sx_thresh_max
    
    cv2.namedWindow('frame')
    
    cv2.createTrackbar('s_th_min','frame',0,255,nothing)
    cv2.createTrackbar('s_th_max','frame',0,255,nothing)
    cv2.createTrackbar('sx_th_min','frame',0,255,nothing)
    cv2.createTrackbar('sx_th_max','frame',0,255,nothing)
        
    ret, img = cap.read()
    
    s=0
    
    while(ret):
        # Capture frame-by-frame
        
        
        s_thresh_min=  cv2.getTrackbarPos('s_th_min','frame')
        s_thresh_max=  cv2.getTrackbarPos('s_th_max','frame')
        sx_thresh_min=  cv2.getTrackbarPos('sx_th_min','frame')
        sx_thresh_max=  cv2.getTrackbarPos('sx_th_max','frame')
        
        if(s==1):
            ret, img = cap.read()
                
        img_draw = pipeline(img)
        
        cv2.imshow('frame', img_draw)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        if cv2.waitKey(5) & 0xFF == ord('s'):
            s=0
        if cv2.waitKey(5) & 0xFF == ord('g'):
            s=1
            
            
def process_video(video_path, white_output):
    
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    
    global loaded, first_frame
    
    first_frame = True    

    loaded = False
    
    #Extend lines in proportion to image
    IMAGE_PROPORTION = True
    
    clip1 = VideoFileClip(video_path)
    white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)
    

left_curve_array = []

right_curve_array = []

def pipeline(img):
    # Câmera undistort
    undistorted = cal_undistort(img)
    
    # Warp Image
    warped = warp(undistorted)            
    
    # Binary Image
    
    global s_thresh_min, s_thresh_max, sx_thresh_min, sx_thresh_max
    
    #s_binary, sxbinary, combined_binary = binary_image(warped, s_thresh=(s_thresh_min, s_thresh_max), sx_thresh=(sx_thresh_min, sx_thresh_max))
    
    s_binary, sxbinary, combined_binary = binary_image(warped, s_thresh=(110, 180), sx_thresh=(20, 100))
    
    #s_binary, sxbinary, combined_binary = binary_image(warped, s_thresh=(170, 200), sx_thresh=(20, 100))
    
    # Cropping edges
    combined_binary[:, -200:] = 0

    combined_binary[:, :200] = 0  
    
    #combined_binary[:1, :] = 0

    combined_binary[690:, :] = 0

    global loaded, left_fit, right_fit, first_frame
        
    # Detect Lines
    # Look ahead
    if(True):
    
        first_frame = False
        leftx, lefty, rightx, righty, left_fit, right_fit, _, _, left_fitx, right_fitx, ploty = detect_lanes(combined_binary)
        
        print('first frame')
        
    else:
    
        leftx, lefty, rightx, righty, _, _, left_fitx, right_fitx, left_fit, right_fit, ploty = detect_lanes_next_frame(combined_binary, left_fit, right_fit)
    
    
    # Measure Angle
    left_curverad, right_curverad = measure_angle(leftx, lefty, rightx, righty, ploty)
    
    left_curve_array.append(left_curverad)
    
    right_curve_array.append(right_curverad)
    
    left_curve_mean = np.array(left_curve_array[-20:]).mean()
    
    right_curve_mean = np.array(right_curve_array[-20:]).mean()
    
    #Reset
    if ((left_curve_mean - right_curve_mean) > 100.0):
        first_frame = True
    
    car_pos = car_position(right_fit[2], left_fit[2], warped.shape[1])
    
    # Draw Lines on Image
    img_draw = draw_lines(combined_binary, undistorted, left_fitx, right_fitx, ploty, left_curve_mean, right_curve_mean, car_pos)
    
    return img_draw

        
if __name__ == "__main__":
    #run_video('project_video.mp4')
    
    process_video('project_video.mp4', 'project_video_output.mp4')
    
    #run_video('challenge_video.mp4')
    
    #run_video('harder_challenge_video.mp4')
    
    
    
    
    

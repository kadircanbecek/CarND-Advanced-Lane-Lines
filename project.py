import glob
import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from lane_finder import LaneFinder
from utils import threshold_colours, threshold_sobelx, combine_binary, transform_n_warp, curvature, undistort_images

calibration_images = glob.glob('camera_cal/calibration*.jpg')

# Calibration points in x and y direction
px = 9
py = 6

# prepare holders for object points in 3D and image points in 2D
obj_points = []
img_points = []

# generate coordinates for object points
objp = np.zeros((py * px, 3), np.float32)
objp[:, :2] = np.mgrid[0:px, 0:py].T.reshape(-1, 2)

for idx, fname in enumerate(calibration_images):

    calib_img = mpimg.imread(fname)
    gray = cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (px, py), None)

    if ret == True:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(calib_img, (9, 6), corners, ret)

img_size = gray.shape[::-1]
plt.imshow(calib_img)
plt.title('Sample Calibration Image')

# Camera calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points,
                                                   img_size[:2], None, None)
np.savetxt('cam_mtx.txt', mtx)
np.savetxt('cam_dist.txt', dist)
# calib_images = glob.glob("camera_cal/calibration*.jpg")

cam_directory = "undistorted_camera_cal"
undistort_images(calibration_images, cam_directory, mtx, dist)

calib_test_images = glob.glob("test_images/*.jpg")
test_directory = "undistorted_test_images"

undistort_images(calib_test_images, test_directory, mtx, dist)

img = mpimg.imread('test_images/test4.jpg')

# for RGB colour space
r_channel = img[:, :, 0]
g_channel = img[:, :, 1]
b_channel = img[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(r_channel, cmap='gray')
ax1.set_title('R Channel', fontsize=20)
ax2.imshow(g_channel, cmap='gray')
ax2.set_title('G Channel', fontsize=20)
ax3.imshow(b_channel, cmap='gray')
ax3.set_title('B Channel', fontsize=20)
f.savefig('output_images/rgb_colourspace')
plt.close(f)

# for HLS colour space
hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
h_channel = hls[:, :, 0]
l_channel = hls[:, :, 1]
s_channel = hls[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(h_channel, cmap='gray')
ax1.set_title('H Channel', fontsize=20)
ax2.imshow(l_channel, cmap='gray')
ax2.set_title('L Channel', fontsize=20)
ax3.imshow(s_channel, cmap='gray')
ax3.set_title('S Channel', fontsize=20)
f.savefig('output_images/hls_colourspace')
plt.close(f)
calib_test_images = glob.glob("test_images/*.jpg")
dirname = "thresholded_test_images"
if not os.path.exists(dirname):
    os.makedirs(dirname)

for _, image_name in enumerate(calib_test_images):
    img = mpimg.imread(image_name)

    c_threshold = threshold_colours(img)
    g_threshold = threshold_sobelx(img)
    combined_binary = combine_binary(c_threshold, g_threshold)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(img, cmap='gray')
    ax1.set_title('Image', fontsize=20)
    ax2.imshow(c_threshold, cmap='gray')
    ax2.set_title('Colour Threshold', fontsize=20)
    ax3.imshow(g_threshold, cmap='gray')
    ax3.set_title('Gradient Threshold', fontsize=20)
    ax4.imshow(combined_binary, cmap='gray')
    ax4.set_title('Combined Binary', fontsize=20)
    fname = "thresholded_" + image_name
    f.savefig(fname)
    plt.close(f)
# Pick points on the image with respect to each image size that
# corresponds to the plane of the road in other to use it to warp the image to a bird's eye view

row, col, _ = img.shape
src_points = np.float32([[0.2 * col, 0.9 * row],
                         [0.45 * col, 0.64 * row],
                         [0.55 * col, 0.64 * row],
                         [0.8 * col, 0.9 * row]])
dst_points = np.float32([[0.2 * col, 0.9 * row],
                         [0.2 * col, 0 * row],
                         [0.8 * col, 0 * row],
                         [0.8 * col, 0.9 * row]])

# obtain perspective transform parameters
M = cv2.getPerspectiveTransform(src_points, dst_points)
Minv = cv2.getPerspectiveTransform(dst_points, src_points)

np.savetxt('M_mat.txt', M)
np.savetxt('Minv_mat.txt', Minv)

test_im = mpimg.imread('test_images/straight_lines1.jpg')
test_im = cv2.undistort(test_im, mtx, dist, None, mtx)
drawn, warped = transform_n_warp(test_im, M, src_points, thresh=True, line=True)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(drawn, cmap='gray')
ax1.set_title('Unwarped Image', fontsize=20)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Birds Eye View Binary', fontsize=20)
f.savefig("output_images/bird_eye_view.jpg")
plt.close(f)

histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)

lower_half = warped[int(warped.shape[0] / 2):, :]  # lower half of image

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(lower_half, cmap='gray')
ax1.set_title('Lower Half', fontsize=20)
ax2.plot(histogram)
ax2.set_title('Histogram', fontsize=20)

f.savefig("output_images/hist_lower_half.jpg")
plt.close(f)

# Create an output image to draw on and  visualize the result
out_img = np.dstack((warped, warped, warped)) * 255

# convert image to integer arrays
out_img = out_img.astype(np.int8)

# Find the peak of the left and right halves of the histogram and those are the starting point for lan
midpoint = np.int(histogram.shape[0] / 2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Number of sliding windows
nwindows = 10

# Height of windows
window_height = np.int(warped.shape[0] / nwindows)

# Identify the x and y positions of all nonzero pixels in the image
nonzero = warped.nonzero()
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
    win_y_low = warped.shape[0] - (window + 1) * window_height
    win_y_high = warped.shape[0] - window * window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin

    # Draw the windows on the visualization image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 5)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 5)

    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
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
ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

# change the colour of nonzero pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [100, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 100]
out_img = out_img * 255
out_img = out_img.astype(np.uint8)
plt.imshow(out_img)
plt.show()
mpimg.imsave("output_images/sliding_window.jpg", out_img)


def draw_curves(image, pts):
    img = image.copy()
    img = cv2.polylines(img, [pts], True, (255, 0, 0), thickness=5)
    return img


test_image = test_im
test_image = transform_n_warp(test_image, M, src_points, thresh=False, line=False)[1]
left_curves = np.asarray([[left_fitx[i], ploty[i]] for i in range(warped.shape[0])]).astype(np.int32)
right_curves = np.asarray([[right_fitx[i], ploty[i]] for i in range(warped.shape[0])]).astype(np.int32)
img = draw_curves(test_image, left_curves)
img = draw_curves(img, right_curves)
plt.imshow(img)
plt.show()
mpimg.imsave("output_images/straight_lines_drawn.jpg", img)

# Skip the sliding windows step once you know where the lines are
binary_warped = transform_n_warp(test_im.copy(), M, src_points, thresh=True)[1]

# Assume you now have a new warped binary image
# from the next frame of video (also called "binary_warped")
# It's now much easier to find line pixels!
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])

# margin = 100
left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
        nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
        nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds]
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)

# Generate x and y values for plotting
ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

# Create an image to draw on and an image to show the selection window

out_img = np.dstack((binary_warped, binary_warped, binary_warped))
window_img = np.zeros_like(out_img)

# # # convert image to integer arrays
out_img = out_img.astype(np.int8)
window_img = window_img.astype(np.int8)

# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [100, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 100]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

# out_img = out_img.astype(np.int8)
result = cv2.addWeighted(out_img, 1, window_img, 0.25, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)

plt.show()
mpimg.imsave("output_images/next_lines.jpg", result.astype(np.uint8))

ym_per_pix = 30 / 720.0  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700.0  # meters per pixel in x dimension

left_curve_rad = curvature(left_fitx, ploty, xm_per_pix, ym_per_pix)
right_curve_rad = curvature(right_fitx, ploty, xm_per_pix, ym_per_pix)

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))


def inverse_warp(image, Minv):
    # Draw the lane onto the warped blank image
    cv2.fillPoly(image, np.int_([pts]), (0, 255, 0))
    image = np.uint8(image)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(image, Minv, image.shape[:2][::-1])
    new_warp = np.uint8(new_warp)
    return new_warp


new_warp = inverse_warp(result, Minv)

# Combine the result with the original image
final_result = cv2.addWeighted(test_im, 1, new_warp, 0.5, 0)
plt.imshow(final_result)
plt.show()
mpimg.imsave("./output_images/final.jpg", final_result.astype(np.uint8))
# left line intercept on x axis
left_intercept = left_fit[0] * img_size[1] ** 2 + left_fit[1] * img_size[1] + left_fit[2]
# right line intercept on x axis
right_intercept = right_fit[0] * img_size[1] ** 2 + right_fit[1] * img_size[1] + right_fit[2]

# calculate lane midpoint
lane_mid = (left_intercept + right_intercept) / 2.0

car_off = (lane_mid - img_size[0] / 2.0) * xm_per_pix


def display_on_frame(image, left_curve_rad, right_curve_rad, car_off):
    # create display texts on image
    font = cv2.FONT_HERSHEY_COMPLEX
    curve_disp_1 = 'Curvature of :'
    curve_disp_2 = 'Left Radius = ' + str(np.round(left_curve_rad, 2)) + 'm'
    curve_disp_3 = 'Right Radius = ' + str(np.round(right_curve_rad, 2)) + 'm'
    off_disp_txt = 'Car off track by ' + str(np.round(car_off, 2)) + 'm'

    cv2.putText(final_result, curve_disp_1, (30, 50), font, 1.5, (255, 255, 255), 2)
    cv2.putText(final_result, curve_disp_2, (30, 100), font, 1.5, (255, 255, 255), 2)
    cv2.putText(final_result, curve_disp_3, (30, 150), font, 1.5, (255, 255, 255), 2)
    cv2.putText(final_result, off_disp_txt, (30, 200), font, 1.5, (255, 255, 255), 2)

    return image


final_result = display_on_frame(final_result, left_curve_rad=left_curve_rad, right_curve_rad=right_curve_rad,
                                car_off=car_off)
plt.imshow(final_result)
plt.show()
plt.imsave('output_images/final_result.png', final_result)

from moviepy.editor import VideoFileClip

frame_line_finder = LaneFinder()
vid_output = 'project_solved.mp4'
clip_source = VideoFileClip("project_video.mp4")
vid_clip = clip_source.fl_image(frame_line_finder.Lane_find)
vid_clip.write_videofile(vid_output, audio=False)

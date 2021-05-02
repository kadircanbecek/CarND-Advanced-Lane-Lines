import os

import cv2
import numpy as np
from matplotlib import image as mpimg, pyplot as plt


def threshold_sobelx(img):
    """
    Return the gradient threshold S channel image

    """
    s_thresh = [170, 255]
    sx_thresh = [25, 200]
    img = np.copy(img)
    # Convert to HLS color space and separate the L,S,R channels
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)

    s_channel = hls[:, :, 2]

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    sobelx_binary = sxbinary

    return sobelx_binary


def bin_it(image, threshold):
    """
    converts a single channeled image to a binary image,
    using upper and lower threshold
    """
    assert len(image.shape) == 2

    output_bin = np.zeros_like(image)
    output_bin[(image >= threshold[0]) & (image <= threshold[1])] = 1
    return output_bin


def threshold_colours(image):
    """
    Return binary image from thresholding colour channels

    """
    # convert image to hls colour space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)

    # binary threshold values
    bin_thresh = [20, 255]

    # rgb thresholding for yellow
    lower = np.array([225, 180, 0], dtype="uint8")
    upper = np.array([255, 255, 170], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    rgb_y = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    rgb_y = cv2.cvtColor(rgb_y, cv2.COLOR_RGB2GRAY)
    rgb_y = bin_it(rgb_y, bin_thresh)

    # hls thresholding for yellow
    lower = np.array([20, 120, 80], dtype="uint8")
    upper = np.array([45, 200, 255], dtype="uint8")
    mask = cv2.inRange(hls, lower, upper)
    hls_y = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_HLS2RGB)
    hls_y = cv2.cvtColor(hls_y, cv2.COLOR_RGB2GRAY)
    hls_y = bin_it(hls_y, bin_thresh)

    # rgb thresholding for white
    lower = np.array([100, 100, 200], dtype="uint8")
    upper = np.array([255, 255, 255], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    rgb_w = cv2.bitwise_and(image, image, mask=mask).astype(np.uint8)
    rgb_w = cv2.cvtColor(rgb_w, cv2.COLOR_RGB2GRAY)
    rgb_w = bin_it(rgb_w, bin_thresh)

    im_bin = np.zeros_like(hls_y)
    im_bin[(rgb_y == 1) | (rgb_w == 1) | (hls_y == 1)] = 1

    return im_bin


def combine_binary(img1, img2):
    combined_binary = np.zeros_like(img1)
    combined_binary[(img1 == 1) | (img2 == 1)] = 1
    return combined_binary


def thresh_bin(image):
    img1 = threshold_colours(image)
    img2 = threshold_sobelx(image)
    c_img = combine_binary(img1, img2)
    return c_img


# Perspective transformation of the image, or the threshold image, along with highlighting the transforming region
def transform_n_warp(image, M, src_points, thresh=False, line=True):
    img = image.copy()
    pts = src_points.astype(np.int32)
    drawn = cv2.polylines(img, [pts], True, (255, 0, 0)) if line else img
    # image size
    img_size = image.shape[:2][::-1]

    # convert to coloured binary image
    image = thresh_bin(image) if thresh else image

    # warp image
    image = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

    return drawn, image


def curvature(x, y, xm, ym):
    # Define y-value where we want radius of curvature, corresponding to the bottom of the image
    y_eval = np.max(y)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y * ym, x * xm, 2)
    curvature = ((1 + (2 * fit_cr[0] * y_eval * ym + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    return curvature


def undistort_images(calib_test_images, test_directory, mtx, dist):
    if not os.path.exists(test_directory):
        os.makedirs(test_directory)
    for idx, image_name in enumerate(calib_test_images):
        img = mpimg.imread(image_name)

        # undistort image
        undist = cv2.undistort(img, mtx, dist, None, mtx)

        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 18))
        f.tight_layout()

        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=30)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=30)
        fname = 'undistorted_' + image_name
        f.savefig(fname)
        plt.close(f)

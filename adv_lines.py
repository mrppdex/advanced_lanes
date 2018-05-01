import numpy as np
from glob import glob
from moviepy.editor import VideoFileClip
import imageio
import cv2


imageio.plugins.ffmpeg.download()

NSEGMENTS = 5 # number of slices of the image
EPSILON = 120 # width of the green rectangle

def calibrate_camera(nx=9, ny=6, calibration_dir='camera_cal/'):
    '''
    Calibrates the camera.
    
    Inputs:
    nx, ny - size of the chessboard
    calibration_dir - directory with distorted chessboard photos
    
    Outputs:
    mtx - camera matrix
    dist - distortion
    '''

    if not nx or not ny:
        nx, ny = (9, 6)
        
    # get the list of the calibration images
    cal_files = glob(calibration_dir + '*.jpg')
    # open and convert them to gray
    cal_images = [cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY) for image in cal_files]
    # initialize object points matrix
    obj_points = np.zeros((nx*ny, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    cimg_points = []
    cobj_points = []
    # perform findChessboardCorner function and add the results to
    # the list of valid chessboard corners if successful
    for img in cal_images:
        ret, corners = cv2.findChessboardCorners(img, (nx, ny), None)
        if ret:
            cimg_points.append(corners)
            cobj_points.append(obj_points)
    # calculate camera and distortion matrices using detected corners
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( \
    cobj_points, cimg_points, cal_images[0].shape[::-1], None, None)
    # returns ret(unused), camera matrix, distortion matrix
    return ret, mtx, dist

def bin_thresholding(img, thresholds):
    '''
    Performs binary thresholding on the image.
    
    Input:
    thresholds - list of ranges of the image values to be assigned value 1.
    i.e. [[30, 70], [200, 255]]
    
    Output:
    bin_img - binarized image
    '''
    bin_img = np.zeros_like(img)
    for i in range(len(thresholds)):
        left_lim = thresholds[i][0]
        right_lim = thresholds[i][1]
        bin_img[(img >= left_lim) & (img <= right_lim)] = 1
    return bin_img

def road_curvature(pl, pr, y_eval, left_lane, right_lane):
    '''
    Calculates road curvature and vehicle position in relation to the
    center of the road.
    
    Inputs:
    pl - (list) polynomial coefficients of the left lane
    pr - (list) polynomial coefficients of the right lane
    y_eval - 'y' value in relation to which the curvature will be calculated
    left_lane - 'x' position of the bottom of the left lane
    right_lane - 'x' position of the bottom of the right lane
    
    Returns:
    curvature of the road and position of the vehicle in relation to
    the center of the road
    '''
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 45/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/847 # meters per pixel in x dimension
    image_width = 1280

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*pl[0]*y_eval*ym_per_pix + pl[1])**2)**1.5) / np.absolute(2*pl[0])
    right_curverad = ((1 + (2*pr[0]*y_eval*ym_per_pix + pr[1])**2)**1.5) / np.absolute(2*pr[0])
    # Now our radius of curvature is in meters
    middle = (right_lane + left_lane) / 2
    pos = (image_width / 2 - middle) * xm_per_pix
    return (left_curverad + right_curverad)/2., pos

def get_warp_mtx():
    '''
    Calculates perspective warp and unwarp matrices
    '''
    # scr_points determined manually
    src_points = np.float32([[535, 470], [760, 470], [0, 719], [1375, 719]])
    dst_points = np.float32([[0, 0], [1279, 0], [0, 719], [1279, 719]])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    M_rev = cv2.getPerspectiveTransform(dst_points, src_points)

    return M, M_rev



def detect_lane(rgb_image, mtx, dist, M, M_rev, old_p=None):
    '''
    Takes a plain image as an input. Outputs a frame with ROI marked and
    some additional telemetrics.
    
    Inputs:
    rgb_image - an input image in rgb format
    mtx - camera matrix
    dist - camera distortion
    M - perspective warp matrix
    M_rev - inverse perspective warp matrix
    old_p - (list) coefficients of left and right lane sides polynomials
    
    Output:
    Processed Image, bottoms of left and right lane sides positions,
    touple with left and right lane sides polynomials.
    '''
    
    # undistort input image
    rgb_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)

    # r channel of an undistorted image
    r_channel = rgb_image[:, :, 0]

    # scaled sobel x, y operation performed on the Y channel of the Y_CR_CB image
    image3 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCR_CB)
    sob3xy = np.abs(cv2.Sobel(image3[:, :, 0], cv2.CV_64F, 1, 1, ksize=7))
    scaled_sob3xy = np.uint8(255*sob3xy/np.max(sob3xy))

    # image converted to the YUV color space
    image1 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YUV)
    
    # combines all the binarized images using binary 'or'
    final_bin_image = \
    bin_thresholding(r_channel, [[220, 255]]) | \
    bin_thresholding(image1[:, :, 1], [[0, 110]]) |\
    bin_thresholding(scaled_sob3xy, [[20, 180]])

    # warp the perspective
    final_warped = cv2.warpPerspective(\
    final_bin_image, M, rgb_image.shape[:2][::-1], flags=cv2.INTER_LINEAR)

    # detect the bottom of the left and right side of the lane
    histogram = np.sum(final_warped[final_warped.shape[0]//2:], axis=0)
    middle = final_warped.shape[1] // 2
    left_peak = np.argmax(histogram[:middle])
    right_peak = np.argmax(histogram[middle:])+middle

    # store them for later
    new_ret_left = left_peak
    new_ret_right = right_peak

    # the middle of the lane
    middle = (right_peak + left_peak)//2

    # lane width
    lane_width = right_peak - left_peak

    # (x, y) points of the detected left side of the lane
    left_x_pts = []
    left_y_pts = []

    # (x, y) points of the detected right side of the lane
    right_x_pts = []
    right_y_pts = []

	# find all x and y of non zero pixels in the warped binarized image
    nonzero_idx = final_warped.nonzero()
    x_idx = nonzero_idx[1]
    y_idx = nonzero_idx[0]

    # augment warped binarized image to 3 channels
    rgb_warped_bin = np.dstack((np.zeros_like(final_warped), \
    np.zeros_like(final_warped), np.zeros_like(final_warped)))

    # determine image slice height
    segment_height = rgb_image.shape[0] // NSEGMENTS

    MAX_PEAK_DIST = EPSILON     # max distance between prev and next lines x pos
    MIN_PEAK_HIST_COUNT = 10    # min number of pixel to consider new peak as valid
    MAX_PEAKS = 4               # number of hist peaks to check
    WEIGHTED_P = 0.3            # new poly coefficients weights old*(1-wp)+new*wp

    # go through all the slices
    for seg in range(NSEGMENTS, 0, -1):
        # histogram of the slice
        slice_hist = np.sum(final_warped[(seg-1)*segment_height:seg*segment_height, :], axis=0)

        # find MAX_PEAKS max left hist peaks indices and sort them highest first
        new_left_peaks = np.sort(\
        np.argpartition(slice_hist[:middle], -MAX_PEAKS)[-MAX_PEAKS:])[::-1]

        # find MAX_PEAKS max right hist peaks indices and sort them highest first
        new_right_peaks = np.sort(\
        np.argpartition(slice_hist[middle:], -MAX_PEAKS)[-MAX_PEAKS:])[::-1]
        # shift it
        new_right_peaks += middle

        # check the conditions if the new left peak may be considered valid
        found_new_left_peak = False
        for i in range(MAX_PEAKS):
            if len(new_left_peaks) > 0 \
            and np.abs(new_left_peaks[i] - left_peak) < MAX_PEAK_DIST and \
            slice_hist[new_left_peaks[i]] >= MIN_PEAK_HIST_COUNT:
                new_left_peak = new_left_peaks[i]
                found_new_left_peak = True
                break
            new_left_peak = left_peak

        # check the conditions if the new right peak may be considered valid 
        found_new_right_peak = False
        for i in range(MAX_PEAKS):
            if len(new_right_peaks) > 0 and \
            np.abs(new_right_peaks[i] - right_peak) < MAX_PEAK_DIST and \
            slice_hist[new_right_peaks[i]] >= MIN_PEAK_HIST_COUNT:
                new_right_peak = new_right_peaks[i]
                found_new_right_peak = True
                break
            new_right_peak = right_peak

        # if one of the peaks is valid and the other is not
        # calculate the position of invalid peak using valid +/- lane width
        if not found_new_left_peak and found_new_right_peak:
            new_left_peak = right_peak - lane_width

        if not found_new_right_peak and found_new_left_peak:
            new_right_peak = left_peak + lane_width

        middle = (new_right_peak + new_left_peak)//2
        lane_width = new_right_peak - new_left_peak
        left_peak, right_peak = new_left_peak, new_right_peak

        # corners of the rectangles of interest around the peaks
        lx1, lx2 = left_peak - EPSILON, left_peak + EPSILON
        ly1, ly2 = (seg-1)*segment_height, seg*segment_height

        rx1, rx2 = right_peak - EPSILON, right_peak + EPSILON
        ry1, ry2 = ly1, ly2

        # draw ROIs onto the final image
        cv2.rectangle(rgb_warped_bin, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
        cv2.rectangle(rgb_warped_bin, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        # color the lanes
        left_idx = ((x_idx >= left_peak - EPSILON) & (x_idx <= left_peak + EPSILON) & \
        (y_idx >= (seg-1)*segment_height) & (y_idx <= seg*segment_height)).nonzero()[0]

        right_idx = ((x_idx >= right_peak - EPSILON) & (x_idx <= right_peak + EPSILON) & \
        (y_idx >= (seg-1)*segment_height) & (y_idx <= seg*segment_height)).nonzero()[0]

        # add (x, y) points of the left and right side to their lists
        y_pt = seg*segment_height

        left_y_pts.append(y_pt)
        left_x_pts.append(left_peak)

        right_y_pts.append(y_pt)
        right_x_pts.append(right_peak)

        rgb_warped_bin[nonzero_idx[0][left_idx], nonzero_idx[1][left_idx], 2] = 255
        rgb_warped_bin[nonzero_idx[0][right_idx], nonzero_idx[1][right_idx], 0] = 255

    # fit the polynomials of the left and right sides of the lane
    new_pr = np.polyfit(right_y_pts, right_x_pts, 2)
    new_pl = np.polyfit(left_y_pts, left_x_pts, 2)

    # if old polynomials coeffs provided in the argument list, use them
    # to weight the new coefficients. If not calculate them.
    if old_p:
        pr = [old_p[1][i]*(1-WEIGHTED_P) + new_pr[i]*WEIGHTED_P for i in range(len(new_pr))]
        pl = [old_p[0][i]*(1-WEIGHTED_P) + new_pl[i]*WEIGHTED_P for i in range(len(new_pl))]
    else:
        pr = np.polyfit(right_y_pts, right_x_pts, 2)
        pl = np.polyfit(left_y_pts, left_x_pts, 2)

    f_left = np.poly1d(pl)
    f_right = np.poly1d(pr)

    # calculate road curvature and vehicle position and draw them
    road_curv, car_pos = road_curvature(pl, pr, rgb_image.shape[0], \
    new_ret_left, new_ret_right)
    cv2.putText(rgb_image, "Curvature = {:.2f} meters".format(road_curv), \
    (520, 100), \
    cv2.FONT_HERSHEY_SIMPLEX, \
    1, \
    (255, 255, 255), \
    2)

    pos_lr = "left" if car_pos < 0 else "right"
    car_pos = np.abs(car_pos)
    cv2.putText(rgb_image,"Position = {:.2f} meters to the {}".format(car_pos, pos_lr), \
    (520, 150), \
    cv2.FONT_HERSHEY_SIMPLEX, \
    1, \
    (255,255,255), \
    2)

    # calculate the edges of a ROI polygon and draw it
    y = np.linspace(0, 719, 10)
    xr = f_right(y)
    xl = f_left(y)

    left = np.vstack((xl, y)).T
    right = np.vstack((xr, y)).T[::-1]

    pts = np.vstack((left, right)).astype(np.int32)

    mask = np.zeros_like(rgb_image)

    imsk = cv2.fillPoly(mask, [pts], [0, 255, 0])
    unwarp = cv2.warpPerspective(imsk, M_rev, rgb_image.shape[:2][::-1], flags=cv2.INTER_LINEAR)
    final = cv2.addWeighted(rgb_image, 1., unwarp, 0.3, 0.2)

    # create warped binarized colorized lane view and add it to the final image
    SHRINK_FACTOR = 0.3
    pip_img = list((np.array(rgb_image.shape[:2])*SHRINK_FACTOR).astype(np.int32)[::-1])
    thumbnail = cv2.resize(rgb_warped_bin, tuple(pip_img))
    th_h, th_w, _ = thumbnail.shape
    final[20:20+th_h, 20:20+th_w, :] = thumbnail

    return final, new_ret_left, new_ret_right, (pl, pr)


if __name__ == '__main__':
    p = None
    _, mtx, dist = calibrate_camera()
    M, M_rev = get_warp_mtx()

    def process_image(image):
        global p
        global mtx, dist, M, M_rev
        result, last_left, last_right, p = detect_lane(\
        image, mtx, dist, M, M_rev, p)
        return result

    vid_output = 'output/lane_detection.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    vid_clip = clip1.fl_image(process_image)
    vid_clip.write_videofile(vid_output, audio=False)


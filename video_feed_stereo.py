import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from random import randint
import stereo_camera_calibration as scc

# Get estimation scale - meter per second #
def reference_scale(T, Q):
    print(T)
    baseline = np.sqrt(T[0]**2+T[1]**2+T[2]**2)

    baseline_ = 1.0/Q[3][2]
    focal_ = Q[2][3]
    Z = (focal_*baseline_)
    print("baseline1(T):{}, baseline2(Q):{}".format(baseline,baseline_))
    print("Focal_length:{}".format(focal_))
    print("Z-depth:{} ".format(Z))

    return Z

# Measure the velocity with depth #
def measure_velocity_depth_point(bbox_l_prev, bbox_l_curr, bbox_r_prev, bbox_r_curr, fps, depth_prev, depth_curr, Q):
    velocity = []
    depth_l = []
    depth_r = []
    baseline_ = 1.0/Q[3][2]
    focal_ = Q[2][3]
    for l1,l2,r1,r2 in zip(bbox_l_prev, bbox_l_curr, bbox_r_prev, bbox_r_curr):
        prev_depth_l = depth_prev[l1[1], l1[0]]
        curr_depth_l = depth_curr[l2[1], l2[0]]
        u = (l1[0], l1[1], prev_depth_l)
        u_tilde = (l2[0], l2[1], curr_depth_l)

        prev_depth_r = depth_prev[r1[1], r1[0]]
        curr_depth_r = depth_curr[r2[1], r2[0]]
        v = (r1[0], r1[1], prev_depth_r)
        v_tilde = (r2[0], r2[1], curr_depth_r)

        # Point with respect to the left_camera
        x_mean_prev_left = (u[0]+Q[0,3])*(prev_depth_l/focal_)
        y_mean_prev_left = (u[1]+Q[1,3])*(prev_depth_l/focal_)
        z_mean_prev_left = prev_depth_l

        x_mean_curr_left = (u_tilde[0]+Q[0,3])*(curr_depth_l/focal_)
        y_mean_curr_left = (u_tilde[1]+Q[1,3])*(curr_depth_l/focal_)
        z_mean_curr_left = curr_depth_l

        p_prev_left = np.array([x_mean_prev_left, y_mean_prev_left, z_mean_prev_left])
        p_curr_left = np.array([x_mean_curr_left, y_mean_curr_left, z_mean_curr_left])
        """
        # Point with respect to the right_camera
        x_mean_prev_right = v[0]-baseline_*(prev_depth_r/focal_)
        y_mean_prev_right = v[1]*(prev_depth_r/focal_)
        z_mean_prev_right = prev_depth_r

        x_mean_curr_right = v_tilde[0]-baseline_*(curr_depth_r/focal_)
        y_mean_curr_right = v_tilde[1]*(curr_depth_r/focal_)
        z_mean_curr_right = curr_depth_r

        p_prev_right = np.array([x_mean_prev_right, y_mean_prev_right, z_mean_prev_right])
        p_curr_right = np.array([x_mean_curr_right, y_mean_curr_right, z_mean_curr_right])
        p_matrix = np.array([p_prev_right, p_curr_right])
        """
        p_matrix = np.array([p_prev_left, p_curr_left])
        dist = np.linalg.norm(p_matrix)
        #dist = np.sqrt((p_curr_left[0]-p_prev_left[0])**2 + (p_curr_left[2]-p_prev_left[2])**2)
        speed = np.round((dist/1),1)
        velocity.append(speed)

        diff_depth_l = np.round(np.abs(prev_depth_l-curr_depth_l),1)
        diff_depth_r = np.round(np.abs(prev_depth_r-curr_depth_r),1)

        depth_r.append(np.round(curr_depth_r,1))
        depth_l.append(np.round(curr_depth_l,1))

    velocity = np.array(velocity)
    depth_l = np.array(depth_l)
    depth_r = np.array(depth_r)

    return velocity, depth_l, depth_r

# A method to display the velocity an depth #
def display_velocity_depth(velocity_boxes, contour_right, contour_left, orig_img_left, orig_img_right, amount_of_bbox, depth_l, depth_r):
    tot_box = 0
    coordinates_left = []
    coordinates_right = []
    for c in contour_right:
        if tot_box == amount_of_bbox:
            break
        x,y,w,h = cv2.boundingRect(c)
        coordinates_right.append([x,y])
        tot_box += 1

    tot_box = 0
    for c in contour_left:
        if tot_box == amount_of_bbox:
            break
        x,y,w,h = cv2.boundingRect(c)
        coordinates_left.append([x,y])
        tot_box += 1

    for i in range(velocity_boxes.shape[0]):
        x_l = coordinates_left[i][0]
        y_l = coordinates_left[i][1]
        x_r = coordinates_right[i][0]
        y_r = coordinates_right[i][1]
        if i == amount_of_bbox-1:
            break
        if velocity_boxes[i] != 0:
            cv2.putText(orig_img_left, "Velocity:"+str(velocity_boxes[i]), (x_l,y_l-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0))
            cv2.putText(orig_img_right, "Velocity:"+str(velocity_boxes[i]), (x_r,y_r-30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0))
            print("Velocity: {} pix/s".format(velocity_boxes[i]))
        #cv2.putText(orig_img_left, "Velocity:"+str(velocity_boxes[i]), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0))
        #cv2.putText(orig_img_right, "Velocity:"+str(velocity_boxes[i]),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0))
        #print("Object: {}, Speed: {}  ".format(i, velocity_boxes[i]))
    i = 0
    for d1,d2 in zip(depth_l, depth_r):
        x_l = coordinates_left[i][0]
        y_l = coordinates_left[i][1]
        x_r = coordinates_right[i][0]
        y_r = coordinates_right[i][1]
        if d1 != 0:
            cv2.putText(orig_img_left, "Depth:"+str(d1),  (x_l,y_l-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
            #cv2.putText(orig_img_left, "Depth:"+str(d1),  (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
            print("___________________________")
            print("Depth_LEFT: {}".format(d1))
            print("___________________________")
        if d2 != 0:
            cv2.putText(orig_img_right, "Depth:"+str(d2),  (x_r,y_r-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
            #cv2.putText(orig_img_right, "Depth:"+str(d2),  (50,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255))
            print("___________________________")
            print("Depth_RIGHT: {}".format(d2))
            print("___________________________")
        i += 1

# Processing the two frames for each two respective cameras(Left and right camera) with previous and current frame -> 4 frames in total #
# Saving the coordinates of points for the different bounding boxes #
def preprocess_two_frames(left_curr, left_prev, right_curr, right_prev, amount_of_bbox):
    if (left_curr == None or left_prev == None or right_curr == None or right_prev == None):
        return

    bbox_l1 = []
    bbox_l2 = []
    bbox_r1 = []
    bbox_r2 = []

    tot_bbox_1 = 0
    for curr_c in left_curr:
        if tot_bbox_1 == amount_of_bbox:
            break
        x1,y1,w1,h1 = cv2.boundingRect(curr_c)
        bbox_l1.append([x1,y1])
        tot_bbox_1 += 1

    tot_bbox_2 = 0
    for curr_c in left_prev:
        if tot_bbox_2 == amount_of_bbox:
            break
        x2,y2,w2,h2 = cv2.boundingRect(curr_c)
        bbox_l2.append([x2,y2])
        tot_bbox_1 += 1

    tot_bbox_3 = 0
    for curr_c in right_curr:
        if tot_bbox_3 == amount_of_bbox:
            break
        x3,y3,w3,h3 = cv2.boundingRect(curr_c)
        bbox_r1.append([x3,y3])
        tot_bbox_3 += 1

    tot_bbox_4 = 0
    for curr_c in right_prev:
        if tot_bbox_4 == amount_of_bbox:
            break
        x4,y4,w4,h4 = cv2.boundingRect(curr_c)
        bbox_r2.append([x4,y4])
        tot_bbox_4 += 1

    bbox_l1 = np.array(bbox_l1)
    bbox_l2 = np.array(bbox_l2)
    bbox_r1 = np.array(bbox_r1)
    bbox_r2 = np.array(bbox_r2)

    return bbox_l1, bbox_l2, bbox_r1, bbox_r2

# Bounding circle #
def bounding_circle(img_binary, orig_image, radius_threshold):
    contours, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img_bin_ret = 0
    for c in contours:
        # Circles inaddDepthstead of rectangular boxes #
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        if radius < radius_threshold:
            img_bin_ret = orig_image
        else:
            img_bin_ret = cv2.circle(orig_image, center, radius, (0, 0, 255), 2)
            cv2.putText(orig_image, "Object detected", (int(x)+radius,int(y)+radius), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))
    return img_bin_ret, contours

# Bounding box around the contoured area #
def bounding_box(img_binary, orig_image, keypoints, w_bound, h_bound):
    contours, hierachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_bin_ret = orig_image
    tot_iterations = 0
    for c in contours:
        if(tot_iterations == keypoints):
            break
        x,y,w,h = cv2.boundingRect(c)
        if w < w_bound and h < h_bound:
            img_bin_ret = orig_image
        else:
            img_bin_ret = cv2.rectangle(orig_image,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(orig_image, "Object detected", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0))

        tot_iterations += 1
    return img_bin_ret, contours

# Extracting the features by filtering out the backgrounds #
def extract_features_dynamic(image,ksize,upper_r,upper_g,upper_b, lower_r, lower_g, lower_b):
    lower_color_bounds = np.array((lower_b,lower_b,lower_r))
    upper_color_bounds = np.array((upper_b,upper_g,upper_r))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = image & mask_rgb

    #kernel = np.ones((ksize,ksize), np.uint8)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    image = cv2.erode(image, kernel, 1)
    image = cv2.dilate(image, kernel, 1)

    return image

# Extracting the features #
def extract_features(image,ksize,color_filter):
    colors = ['B', 'G', 'R']
    if color_filter == colors[2]: # RED
        lower_color_bounds = np.array((0,0,108))
        upper_color_bounds = np.array((50,70,255))
    elif color_filter == colors[1]: # GREEN
        lower_color_bounds = np.array((0,40,0))
        upper_color_bounds = np.array((50,255,50))
    elif color_filter == colors[0]: # BLUE
        lower_color_bounds = np.array((40,0,0))
        upper_color_bounds = np.array((255,50,50))
    else:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, lower_color_bounds,upper_color_bounds)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    image = image & mask_rgb

    #kernel = np.ones((ksize,ksize), np.uint8)
    kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
    #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.erode(image, kernel, 1)
    image = cv2.dilate(image, kernel, 1)

    return image

# Thresholding any arbritary binary image with OTSU #
def binary_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(src=gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def find_Depth(Q_matrix, disparity):
    eps_max = 1e4
    baseline_ = 1.0/Q_matrix[3][2]
    focal_ = Q_matrix[2][3]
    #disparity = np.where(np.isnan(disparity), np.min(disparity), disparity)
    #disparity = np.where(np.isinf(disparity), np.max(disparity), disparity)
    disparity = np.where(disparity == 0, focal_, disparity)
    #disparity = disparity/16.0
    depth = (focal_ * baseline_)/disparity
    return depth

def open_stereo_camera(cam_index1,cam_index2 ,width, height, delta, keyboxes, num_disp, blockSize, amp_scale=1):
    cap_left = cv2.VideoCapture(cam_index1)
    cap_right = cv2.VideoCapture(cam_index2)
    fps = cap_left.get(cv2.CAP_PROP_FPS)
    fps2 = cap_right.get(cv2.CAP_PROP_FPS)


    scc.getChessBoardImages(cap_left, cap_right)
    retval,camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, R, T, E, F, img_shape = scc.camera_stereoCalibrate(c_size_x=7, c_size_y=7)
    r1,r2,p1,p2, Q = scc.camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, img_shape)
    l_cam_x, l_cam_y, r_cam_x, r_cam_y = scc.undistort_RectifyMap(camMatrix1=camMatrix1, camMatrix2=camMatrix2,
                                                                  distCoeffs1=distCoeffs1, distCoeffs2=distCoeffs2,
                                                                  R1=r1, R2=r2,
                                                                  Projection_1=p1, Projection_2=p2, image_shape=img_shape,amp_factor=amp_scale)
    ret, left_Frame = cap_left.read()
    ret2, right_Frame = cap_right.read()

    remap1, remap2 = scc.image_rectification(orig_img_left=left_Frame, orig_img_right=right_Frame,
                                                map1_x=l_cam_x, map1_y=l_cam_y,
                                                map2_x=r_cam_x, map2_y=r_cam_y)
    cv2.imshow("Rectified", np.hstack((remap1, remap2)))

    Z_normalized = reference_scale(T,Q)

    def nothing(x):
        pass
    cv2.namedWindow('Color_filtration', cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar('Upper R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Upper B','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower R','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower G','Color_filtration',0,255,nothing)
    cv2.createTrackbar('Lower B','Color_filtration',0,255,nothing)
    switch_rect = 'rectangle ON \nrectangle OFF'
    switch_circ = 'circle ON \ncircle OFF'
    switch_record = 'record ON \nrecord OFF'
    cv2.createTrackbar(switch_circ, 'Color_filtration',0,1,nothing)
    cv2.createTrackbar(switch_rect, 'Color_filtration',0,1,nothing)

    prev_frame_right = None
    prev_frame_left = None
    prev_depth = None
    prev_contours_left = None
    prev_contours_right = None

    frame_skipped = 0
    frame_skipped_dis = 0
    first_iteration = 0

    while(True):
        ret, left_Frame = cap_left.read()
        ret2, right_Frame = cap_right.read()
        left_Frame = cv2.resize(left_Frame, (width+(3*delta),height+(4*delta)))
        right_Frame = cv2.resize(right_Frame, (width+(3*delta),height+(4*delta)))

        upper_r = cv2.getTrackbarPos('Upper R', 'Color_filtration')
        upper_g = cv2.getTrackbarPos('Upper G', 'Color_filtration')
        upper_b = cv2.getTrackbarPos('Upper B', 'Color_filtration')
        lower_r = cv2.getTrackbarPos('Lower R', 'Color_filtration')
        lower_g = cv2.getTrackbarPos('Lower G', 'Color_filtration')
        lower_b = cv2.getTrackbarPos('Lower B', 'Color_filtration')
        track_circle = cv2.getTrackbarPos(switch_circ, 'Color_filtration')
        track_rectangle = cv2.getTrackbarPos(switch_rect, 'Color_filtration')
        record = cv2.getTrackbarPos(switch_record, 'Color_filtration')

        stuff1 = extract_features_dynamic(image=left_Frame,ksize=3,
                                              upper_r=upper_r, upper_g=upper_g, upper_b=upper_b,
                                              lower_r=lower_r, lower_g=lower_g, lower_b=lower_b)
        stuff2 = extract_features_dynamic(image=right_Frame,ksize=3,
                                              upper_r=upper_r, upper_g=upper_g, upper_b=upper_b,
                                              lower_r=lower_r, lower_g=lower_g, lower_b=lower_b)

        #stuff1 = extract_features(image=left_Frame, ksize=3, color_filter='R')
        #stuff2 = extract_features(image=right_Frame, ksize=3, color_filter='R')
        cv2.imshow("Filtered_left",stuff1)
        cv2.imshow("Filtered_right",stuff2)
        threshold_bin1 = binary_threshold(stuff1)
        threshold_bin2 = binary_threshold(stuff2)

        cv2.imshow("Thresh1", threshold_bin1)
        cv2.imshow("Thresh2", threshold_bin2)

        stereo = cv2.StereoSGBM_create(minDisparity=1, numDisparities=num_disp, blockSize=blockSize)
        left_Frame_g = cv2.cvtColor(left_Frame, cv2.COLOR_BGR2GRAY)
        right_Frame_g = cv2.cvtColor(right_Frame, cv2.COLOR_BGR2GRAY)
        disparity = stereo.compute(right_Frame, left_Frame)

        #min_disparity = np.min(disparity)
        #max_disparity = np.max(disparity)
        #disparity = np.uint8(255*(disparity - min_disparity)/(max_disparity-min_disparity))
        #cv2.imshow("disparity_REAL", disparity)
        depths = find_Depth(Q_matrix=Q, disparity=disparity)

        if frame_skipped%fps == 0:
            prev_frame_left = left_Frame
            prev_frame_right = right_Frame
            bbox_l2, contours_prev_left = bounding_box(img_binary=threshold_bin1, orig_image=prev_frame_left, keypoints=keyboxes, w_bound=20, h_bound=20)
            bbox_r2, contours_prev_right = bounding_box(img_binary=threshold_bin2, orig_image=prev_frame_right, keypoints=keyboxes, w_bound=20, h_bound=20)
            prev_depths = depths
            prev_contours_left = contours_prev_left
            prev_contours_right = contours_prev_right
            frame_skipped = 0
        frame_skipped += 1

        if track_circle == 1:
            bbox_l_circ, cont_circ_l = bounding_circle(img_binary=threshold_bin1, orig_image=left_Frame, radius_threshold=30)
            bbox_r_circ, cont_circ_r = bounding_circle(img_binary=threshold_bin2, orig_image=right_Frame, radius_threshold=30)
            cv2.imshow('Bounding_Box_Left', left_Frame)
            cv2.imshow('Bounding_Box_Right', right_Frame)

        if track_rectangle == 1:
            bbox_l1, contours_curr_left = bounding_box(img_binary=threshold_bin1, orig_image=left_Frame, keypoints=keyboxes, w_bound=30, h_bound=30)
            bbox_r1, contours_curr_right = bounding_box(img_binary=threshold_bin2, orig_image=right_Frame, keypoints=keyboxes, w_bound=30, h_bound=30)

            b1,b2,b3,b4 = preprocess_two_frames(left_curr=contours_curr_left, left_prev=prev_contours_left,
                                                right_curr=contours_curr_right, right_prev=prev_contours_right, amount_of_bbox=keyboxes)

            velocity_boxes, depth_l, depth_r = measure_velocity_depth_point(bbox_l_prev=b1, bbox_l_curr=b2,
                                                                           bbox_r_prev=b3, bbox_r_curr=b4,
                                                                           fps=fps, depth_curr=depths, depth_prev=prev_depths,
                                                                           Q=Q)

            display_velocity_depth(velocity_boxes=velocity_boxes, amount_of_bbox=keyboxes,
                                 depth_l=depth_l, depth_r=depth_r,
                                contour_right=contours_curr_right, contour_left=contours_curr_left,
                                orig_img_left=left_Frame, orig_img_right=right_Frame)

            cv2.imshow('Bounding_Box_Left',left_Frame)
            cv2.imshow('Bounding_Box_Right',right_Frame)

        if track_circle == 0 and track_rectangle == 0:
            cv2.imshow('Bounding_Box_Left',left_Frame)
            cv2.imshow('Bounding_Box_Right',right_Frame)

        disparity_Adjmap = np.uint8(disparity)
        min_disp = np.max(disparity)
        max_disp = np.min(disparity)
        scale = 255/(max_disp-min_disp)
        disparity_Adjmap = cv2.convertScaleAbs(disparity_Adjmap,disparity_Adjmap ,scale, -min_disp*scale)
        disparity_colormap = cv2.applyColorMap(disparity_Adjmap, cv2.COLORMAP_JET) # COLORMAP_BONE, COLORMAP_OCEAN

        depths_Adjmap = np.uint8(depths)
        min_disp = np.max(depths)
        max_disp = np.min(depths)
        scale = 255/(max_disp-min_disp)
        depths_Adjmap = cv2.convertScaleAbs(depths_Adjmap, depths_Adjmap,scale, -min_disp*scale)
        depths_colormap = cv2.applyColorMap(depths_Adjmap, cv2.COLORMAP_JET) # COLORMAP_BONE, COLORMAP_OCEAN

        cv2.imshow("Disparity", disparity_colormap)
        cv2.imshow("Depths", depths_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

"""
def main():
    open_stereo_camera(cam_index1=0, cam_index2=2, width=640, height=480, delta=0, keyboxes=10)

if __name__ == '__main__':
    main()
"""

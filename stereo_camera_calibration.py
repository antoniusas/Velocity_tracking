import numpy as np
import cv2
import glob
import os

def getChessBoardImages(cap, cap2):
    if (len(os.listdir('images')) != 0):
        print("Folder is not empty -- Existing images --")
        return

    i = 0
    while (True):
        ret, left_Frame = cap.read()
        ret2, right_Frame = cap2.read()
        left_Frame = cv2.cvtColor(left_Frame, cv2.COLOR_BGR2GRAY)
        right_Frame = cv2.cvtColor(right_Frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Left-camera", left_Frame)
        cv2.imshow("Right-camera", right_Frame)
        k = cv2.waitKey(1)
        if k == 32: # Space
            cv2.imwrite('images/left_'+str(i)+'.png', left_Frame)
            cv2.imwrite('images/right_'+str(i)+'.png', right_Frame)
            i += 1
            print("Picture: {}".format(i))
        elif k == 27: # Esc
            break

def camera_stereoCalibrate(c_size_x, c_size_y):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((c_size_x*c_size_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:c_size_x,0:c_size_y].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpointsL = [] # 3d point in real world space
    imgpointsL = [] # 2d points in image plane.
    objpointsR = []
    imgpointsR = []

    images_left = glob.glob('images/left*.png')
    images_left.sort()
    images_right = glob.glob('images/right*.png')
    images_right.sort()
    found_matches = 0
    for fname1, fname2 in zip(images_left, images_right):
        imgL = cv2.imread(fname1)
        grayL = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
        ret_L, cornersL = cv2.findChessboardCorners(grayL,(c_size_x, c_size_y),None)

        imgR = cv2.imread(fname2)
        grayR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
        ret_R, cornersR = cv2.findChessboardCorners(grayR, (c_size_x,c_size_y),None)
        #print(cornersL)
        if ret_L == True:
            ret_left = cv2.drawChessboardCorners(imgL, (c_size_x,c_size_y), cornersL, ret_L)
            cv2.imshow("Left-camera",ret_left)
            cv2.waitKey(1)
            objpointsL.append(objp)
            cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)
            imgpointsL.append(cornersL)

        if ret_R == True:
            ret_right = cv2.drawChessboardCorners(imgR, (c_size_x,c_size_y), cornersR, ret_R)
            cv2.imshow("Right-camera",ret_right)
            cv2.waitKey(1)
            objpointsR.append(objp)
            cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)
            imgpointsR.append(cornersR)

        if (ret_R and ret_L):
            found_matches += 1

    rt1, m1,d1,r1,t1 = cv2.calibrateCamera(objpointsL, imgpointsL, grayL.shape, None, None)
    rt2, m2,d2,r2,t2 = cv2.calibrateCamera(objpointsR, imgpointsR, grayR.shape, None, None)

    m1_new, roi_1 = cv2.getOptimalNewCameraMatrix(m1, d1, grayL.shape, 1)
    m2_new, roi_2 = cv2.getOptimalNewCameraMatrix(m2, d2, grayR.shape, 1)


    #nL = np.array(imgpointsL).shape[0]
    #nR = np.array(imgpointsR).shape[0]
    #nImg = min(nL, nR)

    termination_criteria_extrinsics =  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    flags = cv2.CALIB_FIX_INTRINSIC

    retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objpointsL[0:found_matches], imgpointsL[0:found_matches], imgpointsR[0:found_matches],
                                                                                                    m1_new, d1, m2_new, d2, grayL.shape,flags=flags, criteria=termination_criteria_extrinsics)
    img_shape = grayL.shape
    print("Reprojection Error Camera_right:", rt1)
    print("Reprojection Error Camera_left:", rt2)
    print("Reprojection Error Stereo:", retval)
    cv2.destroyWindow("Left-camera")
    cv2.destroyWindow("Right-camera")
    return retval,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F, img_shape

def camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, image_shape):
    Rectification_1, Rectification_2, Projection_1, Projection_2, Q, roi1, roi2= cv2.stereoRectify(camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, image_shape, R, T,
                                                      cv2.CALIB_ZERO_DISPARITY, -1, image_shape)

    return Rectification_1, Rectification_2, Projection_1, Projection_2, Q

def undistort_RectifyMap(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R1, R2, Projection_1, Projection_2, image_shape, amp_factor=1):
    image_shape_resize = (image_shape[0]*amp_factor, image_shape[1]*amp_factor)
    left_cam_mapX, left_cam_mapY = cv2.initUndistortRectifyMap(cameraMatrix=camMatrix1, distCoeffs=distCoeffs1,
                                                               R=R1, newCameraMatrix=Projection_1,
                                                               size=image_shape_resize, m1type=cv2.CV_32FC1)

    right_cam_mapX, right_cam_mapY = cv2.initUndistortRectifyMap(cameraMatrix=camMatrix2, distCoeffs=distCoeffs2,
                                                               R=R2, newCameraMatrix=Projection_2,
                                                               size=image_shape_resize, m1type=cv2.CV_32FC1)
    return left_cam_mapX, left_cam_mapY, right_cam_mapX, right_cam_mapY

def image_rectification(orig_img_left, orig_img_right, map1_x, map1_y, map2_x, map2_y):
    remap1 = cv2.remap(src=orig_img_left, map1=map1_x, map2=map1_y, interpolation=cv2.INTER_LINEAR)
    remap2 = cv2.remap(src=orig_img_right, map1=map1_x, map2=map1_y, interpolation=cv2.INTER_LINEAR)

    #cv2.imshow("Undistorted RectifyMap1", remap1)
    #cv2.imshow("Undistorted RectifyMap2", remap2)
    #cv2.waitKey(0)
    #cv2.destroyWindow("Undistorted RectifyMap1")
    #cv2.destroyWindow("Undistorted RectifyMap2")
    return remap1, remap2

"""
def main():
    getChessBoardImages(cam_index1=4, cam_index2=6)
    retval,camMatrix1, distCoeffs1, camMatrix2, distCoeffs2, R, T, E, F, img_shape = camera_stereoCalibrate(c_size_x=7, c_size_y=7)
    r1,r2,p1,p2, disp_depth = camera_stereoRectify(camMatrix1, camMatrix2, distCoeffs1, distCoeffs2, R, T, img_shape)

if __name__ == '__main__':
    main()
"""

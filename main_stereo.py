import numpy as np
import cv2 as cv2
import video_feed_stereo as vfs
def main():
    cam_index1 = 1
    cam_index2 = 2
    cap_left = cv2.VideoCapture(cam_index1)
    cap_right = cv2.VideoCapture(cam_index2)
    ret_left, left_Frame = cap_left.read()
    ret_right, right_Frame = cap_right.read()
    if (ret_left == False) or (ret_right == False):
        print("--CAMERA NOT FOUND--")
        cap_left.release()
        cap_right.release()
        return

    vfs.open_stereo_camera(cam_index1=cam_index1, cam_index2=cam_index2,
                                  width=640, height=480,
                                  delta=0, keyboxes=3,
                                  num_disp=32, blockSize=5, amp_scale=5)
if __name__ == '__main__':
    main()

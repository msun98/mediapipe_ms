#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp
from annotation import * 

import threading
from calculate_rotation import *
from initial import *
from ptz_api import *
import time
from utils import CvFpsCalc


w_calibration, h_calibration = 1527, 833
mid_x, mid_y = w_calibration/2, h_calibration/2
neck_x,neck_y = 0,0 
point_flag = True
lock = threading.Lock()
end_time = 0
r_cx,l_cx=0,0
zoom = False
dt = 1/20
sigmaX, sigmaA = 0.0001, 0.0015
gesture_flag = False
gesture = 0
z=0
cpx, cpy = w_calibration/2,h_calibration/2


F = [ [1, dt, 0.5*dt**2, 0, 0, 0],
      [0, 1, dt, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5*dt**2],
      [0, 0, 0, 0, 1, dt],
      [0, 0, 0, 0, 0, 1]]

H = [ [1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0]]

Q = [ [dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
      [dt**3/2, dt**2, dt, 0, 0, 0],
      [dt**2/2, dt, 1, 0, 0, 0],
      [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
      [0, 0, 0, dt**3/2,dt**2, dt],
      [0, 0, 0, dt**2/2, dt, 1]]

R = [[1,0],[0,1]]


kalman = cv2.KalmanFilter(6, 2)   #  state [ x, vx, ax, y, vy, ay]
kalman.transitionMatrix = np.array(F, np.float32)
kalman.measurementMatrix = np.array(H, np.float32) # System measurement matrix
kalman.processNoiseCov = np.array(Q, np.float32)*sigmaA**2 # System process noise covariance
kalman.measurementNoiseCov = np.array(R, np.float32) * sigmaX**1

neck_x, neck_y = w_calibration/2, h_calibration/2

kalman.statePre = np.array([[neck_x], [0], [0], [neck_y],[0],[0]], np.float32)
kalman.statePost = np.array([[neck_x], [0], [0], [neck_y],[0],[0]], np.float32) 

# cpx, cpy = w_calibration/2, h_calibration/2

# kalman.statePre = np.array([[cpx], [0], [0], [cpy],[0],[0]], np.float32)
# kalman.statePost = np.array([[cpx], [0], [0], [cpy],[0],[0]], np.float32) 

ges = 1

def goto_human():
    global point_flag,cpx, cpy, w_calibration,h_calibration,neck_x,neck_y,end_time,zoom,gesture_flag,gesture,z,ges
    angle_of_x_old, angle_of_y_old = 0, 0
    pp, tp = 0, 0
    pp_old,tp_old = 0,0
    pan = 0
    predict_x_old = 0
    start = time.time()
    a = 1
    zoom_position = 0 
    neck = 0
    run = True
    # ges = 2
    parall = False
    horizon_aov_max = 56.4
    horizon_aov_min = 3.1

    while True:
        lock.acquire()
        if point_flag:
            # print('working')
            if zoom:
                if z == 1:
                    move('in',10)
                    # print('zoom in')
                    time.sleep(1)
                    stop('in')


                elif z == 2:
                    move('out',10)
                    time.sleep(1)

                    # print('zoom out')
                    stop('out')

                # zoom = False
                r_cx,l_cy = 0,0
                zoom_position = int(get_position()[2])

                z = 3
                run = True
                zoom = False


            if gesture_flag:
                run = False
                # panpos = int((get_position()[0])/100)
                th_w = float(horizon_aov_max-(horizon_aov_max-horizon_aov_min)/65535*zoom_position)
                # print(th_w)
                if ges == 1:
                    moveTo(int(pan*100),int(tilt*100),0,0)
                    ges = 2

                if gesture == 1:
                    alpha = int(th_w * 1/5)
                    moveTo(pan+int(alpha*100),int(tilt*100),10,0)
                    # ?????????. ?????? ????????? ?????????.(pan ??? ??????)
                    print("?????????")
                    neck = alpha
                    parall = True

                if gesture == 2:
                    alpha = -int(th_w * 1/5)
                    # pan_alpha=360+alpha
                    moveTo(pan+int((360+alpha)*100),int(tilt*100),10,0)
                    print("??????")
                    neck = alpha
                    parall = True

                if gesture == 4:
                    # run = False
                    # moveTo(int((pan+neck)*100),int(tilt*100),5,0)
                    # print(parall)
                    parall = True
                    gesture = 3

                run = True
                gesture_flag = False

            if run:
                # if parall:
                #     neck_x += neck
                
                pan, tilt = np.rad2deg(calculate_alpha(neck_x,zoom_position)),np.rad2deg(calculate_beta(neck_y,zoom_position))
                # TO CALCULATE OF MOTOR SPEED
                if parall:
                    if gesture == 4:
                        pan -= neck
                        parall = False
                        # pass
                    else:
                        pan += neck

                if a == 1:
                    moveTo(int(pan*100),int(tilt*100),0,0)
                    time.sleep(1)
                    a = 2

                #
                pp, tp = int(np.abs(pan)), int(np.abs(tilt)*0.3)
                # print('pp,tp:',pp,tp)

                if np.abs(pp_old - pp) != 0 or np.abs(tp_old - tp) != 0:

                    if pan < 0:
                        # print('\npan < 0\n')
                        if tilt > 0:
                            move_pan_tilt('right', 'up', pp, tp)

                         # ??? ?????? ????????? ????????? ???????????? ?????? stop
                        elif np.abs(pan) < 3:
                            # print('neck is on center')
                            moveTo(int(pan*100),int(tilt*100),0,0)

                        else:
                            move_pan_tilt('right', 'down', pp, tp)


                    elif pan > 0:
                        # print('\npan > 0\n')
                        if tilt > 0:
                            move_pan_tilt('left', 'up',pp, tp)

                        # ??? ?????? ????????? ????????? ???????????? ?????? stop
                        elif np.abs(pan) < 3:
                            # print('neck is on center')
                            moveTo(int(pan*100),int(tilt*100),0,0)
                            

                        else:
                            move_pan_tilt('left', 'down', pp, tp)


                    elif np.abs(pan - w_calibration/2) < 10:
                        moveTo(int(pan*100), int(tilt*100), 0, 5)
                        # print('\nstop\n')
                
                    # print(pp,tp)
                    pp_old, tp_old = pp, tp
                        # print(pp_old,tp_old)

                else:
                    print('')
                    # a=1
    

        lock.release()


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    # ????????? ????????? (0 : Lite 1 : Full 2 : Heavy)
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=0)
    # ?????? ?????? ?????? ?????? ??? (????????? : 0.5)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    # ?????? ?????? ?????? ?????? ??? (????????? : 0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)

    # # ?????? ???????????? ?????? ????????? ?????? (????????? : ???????????? ??????)
    parser.add_argument('--use_brect', action='store_true')
    # parser.add_argument('--use_brect', action='store_false')


    args = parser.parse_args()

    return args



if __name__ == '__main__':
    # main()
    # ???????????? #################################################################
    args = get_args()
    cap = initialize() # ????????? ?????????.
    on_screen_display()
    i = 1
    r_cx,l_cx = 0,0
    r_cy,l_cy = 0,0
    cpx, cpy = w_calibration/2,h_calibration/2
    length_old = 0
    length = 0

    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fps = 20
    # out = cv2.VideoWriter('save_video/?????? ??????.avi', fourcc, fps, (1527, 833))

    # upper_body_only = args.upper_body_only
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # ????????? ?????? ###############################################################
    ret_val, image = cap.read()
    video = calibration(image, w_calibration, h_calibration)

    # ???????????? #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    initial_position = get_position()
    if initial_position != (0, 0, 0):
        goto_origin(0,100,0,100)
        # http://192.168.0.9/cgi-bin/control/zf_control.cgi?id=admin&passwd=admin&action=setzfmove&zoom=out&zoompeed=50
        time.sleep(3)
        stop('out')
        time.sleep(3)

    human = threading.Thread(target=goto_human)
    human.daemon = True  # ???????????? ????????? ?????? ??????.
    human.start()

    # FPS?????? ?????? ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    start = time.time()
    while True:
        if cap.isOpened():
            display_fps = cvFpsCalc.get()

            start_camera = time.time()

            # ????????? ?????? #####################################################
            ret, image = cap.read()
            if not ret:
                break

            if i == 1:
                goto_origin(pp,ps,tp,ts)
                # stop('out')
                timer_out = time.time()-start
                if timer_out > 5:
                    stop('out')
                    i = 2

            image = calibration(image, w_calibration, h_calibration)
            debug_image = copy.deepcopy(image)

            # ?????? ?????? #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Pose ###############################################################
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                # ?????? ???????????? ??????
                # ????????? 
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                pose = debug_image
                right_sholder_x, right_sholder_y, left_sholder_x, left_sholder_y = pose[1], pose[2], pose[3], pose[4]

                # ??? ?????? 
                point_flag = True
                # print(point_flag)
                neck_x,neck_y = int((pose[1]+pose[3])/2) ,int((pose[2]+pose[4])/2) 
                end_time = time.time()-start_camera
                # print(end_time)
                debug_image = cv.circle(pose[0], (neck_x, neck_y), 5, (0, 0, 255), 2)


            current_prediction = kalman.predict() # start kalman filter <predict>
            cpx, cpy = int(current_prediction[0]), int(current_prediction[3])
            # debug_image = cv2.circle(debug_image, (cpx, cpy), 8, (255, 0, 0), -1)
            if np.abs(cpy-neck_y) > 500:
                cpy = int(h_calibration/2)
            current_measurement = np.array([neck_x, neck_y], np.float32)
            kalman.correct(current_measurement)
            # print(current_measurement)
            # cpx, cpy = int(current_measurement[0]), int(current_measurement[1])
            # print(cpx,cpy)
            # debug_image = cv2.circle(debug_image, (cpx, cpy), 8, (255, 0, 0), -1)

            
            # Hands ###############################################################
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks

            # ??????
            if left_hand_landmarks is not None:
                # ????????? ?????? ?????? 
                l_cx, l_cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # ?????????????????? ??????
                debug_image = draw_hands_landmarks(
                    debug_image,
                    l_cx,
                    l_cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )
                

                if np.abs(neck_y-l_cy) < 50 :
                    if np.abs((debug_image[2][1] + debug_image[5][1]) - (debug_image[3][1] + debug_image[4][1])) < 10:
                        # (?????? + ??????) - (?????? + ??????)
                        gesture_flag = True
                        # ges = 1
                        # ??????????????? ?????? ??????.
                        if debug_image[1][0]-debug_image[2][0] < 0:
                            # ????????? ???????????? ????????? ????????? ?????????
                            # ?????????
                            # if np.abs(debug_image[1][1]-l_cy) < 30:
                                # gesture = 4
                            # else:
                            gesture = 1

                        else:
                            gesture = 2
                    else:
                        if debug_image[1][0]-debug_image[5][0] < 0:
                        # ????????? ???????????? ????????? ??? ?????????.
                            print('?????????')
                            gesture = 4
                        else:
                            pass
                debug_image = debug_image[0]

            if gesture == 1:

                debug_image = cv.putText(debug_image, "right", (200, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                gesture = 3

            if gesture == 2:

                debug_image = cv.putText(debug_image, "left", (200, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                gesture = 3

            if gesture == 4:

                debug_image = cv.putText(debug_image, "center", (200, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
                gesture = 3

            # ????????? 
            if right_hand_landmarks is not None:
                # ????????? ?????? ?????? 
                r_cx, r_cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # ?????????????????? ??????
                # brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # ????????? 
                debug_image = draw_hands_landmarks(
                    debug_image,
                    r_cx,
                    r_cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )
                thumb,index,middel, ring, pinky = debug_image[1][1], debug_image[2][1], debug_image[3][1], debug_image[4][1],debug_image[5][1]
                debug_image = debug_image[0]


            # print(r_cx,l_cx)
            # if 
            # debug_image=cv.circle(debug_image, (int((r_cx+l_cx)/2), int((r_cy+l_cy)/2)),5, (0, 0, 255), 2)
            if np.abs(neck_y-r_cy) < 20 and np.abs(neck_y-l_cy) < 20:
                #???????????? ???????????? ????????????! ??????????????? ????????? ???????????? ??????.
                if r_cx*r_cy*l_cx*l_cy != 0: 
                    # if np.abs((index + pinky) - (middel + ring)) > 10:
                        # print(np.abs((index + pinky) - (middel + ring)))
                        # ?????? ?????? ????????? ???????????? y??? ??? ?????? ????????? ??? 0??? ????????? ????????????.
                    debug_image = cv.line(debug_image,(r_cx,r_cy),(l_cx,l_cy),(0, 255, 0), 2)

                    zoom = True
                    if left_sholder_x < l_cx & r_cx < right_sholder_x:   
                    # if np.abs(right_sholder_x - left_sholder_x ) > np.abs(r_cx - l_cx):
                        # print(right_sholder_x, r_cx, l_cx, left_sholder_x)
                        z = 1
                        debug_image = cv.putText(debug_image, "ZOOM IN", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                    else:
                        z = 2
                        debug_image = cv.putText(debug_image, "ZOOM OUT", (10, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                # length_old = length

            # cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
            #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            # ?????? ??????  #############################################################
            # out.write(debug_image)
            debug_image = cv.imshow('MediaPipe Holistic', debug_image)
            # out.write(debug_image)


            # ????????? (ESC?????????) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                stop('out')
                time.sleep(0.3)
                goto_origin(0,100,0,100)
                time.sleep(0.3)
                break

            # # ?????? ??????  #############################################################
            # cv.imshow('MediaPipe Holistic', debug_image)

    # out.release()
    cap.release()
    cv.destroyAllWindows()

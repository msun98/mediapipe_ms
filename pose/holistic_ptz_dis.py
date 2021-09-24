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
neck_x,neck_y = 0,0 
point_flag = True
lock = threading.Lock()
end_time = 0
r_cx,l_cx=0,0
zoom = False
gesture_flag = False
gesture = 0
z=0
# ges = 2

def goto_human():
    global point_flag,cpx, cpy,neck_x,neck_y,end_time,zoom,z,gesture_flag,gesture
    angle_of_x_old, angle_of_y_old = 0, 0
    pp, tp = 0, 0
    pp_old,tp_old = 0,0
    pan = 0
    predict_x_old = 0
    start = time.time()
    a = 1
    zoom_position = 0 
    neck=0
    run = True
    ges = 2
    parall = False

    while True:
        lock.acquire()
        if point_flag:
            # print('working')
            if zoom:
                if z == 1:
                    move('in',10)
                    # print('zoom in')
                    stop('in')


                elif z == 2:
                    move('out',10)
                    # print('zoom out')
                    stop('out')

                # zoom = False
                r_cx,l_cy = 0,0
                zoom_position = int(get_position()[2])
                # r_cx,r_cy = 0,0
                z = 3
                run = True
                zoom = False


            if gesture_flag:
                run = False
                if ges == 1:
                    moveTo(int(pan*100),int(tilt*100),0,0)
                    ges = 2

                if gesture == 1:
                    # run = False
                    print("오른쪽")
                    # run = False
                    move('left',30)
                    stop('left')
                    parall = True

                elif gesture == 2:
                    # run = False
                    print("왼쪽")
                    # run = False
                    move('right',30)
                    stop('right')
                    parall = True

                elif gesture == 4:
                    # run = False
                    parall = False

                neck = np.rad2deg(calculate_alpha(neck_x,zoom_position))
                # print(neck)
                # parall = True
                run = True
                gesture_flag = False

            if run:
                if parall:
                    neck_x += neck

                pan, tilt = np.rad2deg(calculate_alpha(neck_x,zoom_position)),np.rad2deg(calculate_beta(neck_y,zoom_position))
                # TO CALCULATE OF MOTOR SPEED
                if parall:
                    pan += -neck

                if a == 1:
                    moveTo(int(pan*100),int(tilt*100),0,0)
                    time.sleep(1)
                    a = 2

                #
                pp, tp = int(np.abs(pan)*1.3), int(np.abs(tilt))
                # print('pp,tp:',pp,tp)

                if np.abs(pp_old - pp) != 0 or np.abs(tp_old - tp) != 0:

                    if pan < 0:
                        # print('\npan < 0\n')
                        if tilt > 0:
                            move_pan_tilt('right', 'up', pp, tp)

                         # 팬 값이 적당히 중앙에 위치하면 모터 stop
                        elif np.abs(pan) < 3:
                            # print('neck is on center')
                            moveTo(int(pan*100),int(tilt*100),0,0)

                        else:
                            move_pan_tilt('right', 'down', pp, tp)


                    elif pan > 0:
                        # print('\npan > 0\n')
                        if tilt > 0:
                            move_pan_tilt('left', 'up',pp, tp)

                        # 팬 값이 적당히 중앙에 위치하면 모터 stop
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
                    angle_of_x_old, angle_of_y_old = pan, tilt
                        # print(pp_old,tp_old)

                else:
                    print('')
                    # a=1
    

        lock.release()


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    # 모델의 복잡도 (0 : Lite 1 : Full 2 : Heavy)
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=0)
    # 감지 신뢰 값의 임계 값 (기본값 : 0.5)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    # 추적 신뢰 값의 임계 값 (기본값 : 0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)

    # # 경계 사각형을 그릴 것인지 여부 (기본값 : 지정되지 않음)
    parser.add_argument('--use_brect', action='store_true')
    # parser.add_argument('--use_brect', action='store_false')


    args = parser.parse_args()

    return args



if __name__ == '__main__':
    # main()
    # 인자분석 #################################################################
    args = get_args()
    cap = initialize() # 화면을 받아옴.
    on_screen_display()
    i = 1
    hand=3
    r_cx,l_cx = 0,0
    r_cy,l_cy = 0,0
    length_old = 0
    length = 0

    # upper_body_only = args.upper_body_only
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # 카메라 준비 ###############################################################
    ret_val, image = cap.read()
    video = calibration(image, w_calibration, h_calibration)

    # 모델로드 #############################################################
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
        # stop('out')
        time.sleep(3)

    human = threading.Thread(target=goto_human)
    human.daemon = True  # 프로그램 종료시 즉시 종료.
    human.start()

    # FPS측정 모듈 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        if cap.isOpened():
            display_fps = cvFpsCalc.get()

            start_camera = time.time()

            # 카메라 캡쳐 #####################################################
            ret, image = cap.read()
            if not ret:
                break

            if i == 1:
                goto_origin(pp,ps,tp,ts)
                # stop('out')
                time.sleep(0.3)
                stop('out')
                i = 2

            image = calibration(image, w_calibration, h_calibration)
            debug_image = copy.deepcopy(image)

            # 감지 실시 #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Pose ###############################################################
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:

                # 그리기 
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                pose = debug_image
                right_sholder_x, right_sholder_y, left_sholder_x, left_sholder_y = pose[1], pose[2], pose[3], pose[4]

            #     # 목 위치 
            #     point_flag = True
            #     # print(point_flag)
                neck_x,neck_y = int((pose[1]+pose[3])/2) ,int((pose[2]+pose[4])/2) 
                end_time = time.time()-start_camera
            #     # print(end_time)
                debug_image = cv.circle(pose[0], (neck_x, neck_y), 5, (0, 0, 255), 2)
            #     # point_flag = True

            
            # Hands ###############################################################
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks

            # 왼손
            if left_hand_landmarks is not None:
                # 손바닥 중심 계산 
                l_cx, l_cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # 그리기 
                debug_image = draw_hands_landmarks(
                    debug_image,
                    l_cx,
                    l_cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )

                
                debug_image = debug_image[0]

            # 오른손 
            if right_hand_landmarks is not None:
                # 손바닥 중심 계산 
                r_cx, r_cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 그리기 
                debug_image = draw_hands_landmarks(
                    debug_image,
                    r_cx,
                    r_cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )

                if np.abs((debug_image[5][1] + debug_image[2][1]) - (debug_image[3][1] + debug_image[4][1])) < 20:
                    # (검지 + 새끼) - (중지 + 약지)
                    gesture_flag = True
                    # 거울모드라 이게 맞음.
                    if debug_image[1][0]-debug_image[2][0] < 0:
                        # 좌회전
                        if np.abs(debug_image[1][1]-r_cy) < 50:
                            gesture = 4
                        else:
                            gesture = 1

                    else:
                        gesture = 2
                else:
                    print('손펼침')
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


            if np.abs(neck_y-r_cy) < 50 & np.abs(neck_y-l_cy) < 50:
                if r_cx*r_cy*l_cx*l_cy != 0: 
                    debug_image = cv.line(debug_image,(r_cx,r_cy),(l_cx,l_cy),(255, 0, 0), 2)
                    zoom = True

                    if left_sholder_x < r_cx & l_cx < right_sholder_x:      
                        z = 1
                        debug_image = cv.putText(debug_image, "ZOOM IN", (200, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                    else:
                        z = 2
                        debug_image = cv.putText(debug_image, "ZOOM OUT", (200, 30),
                        cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

                length_old = length

            # debug_image = cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
            #            cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            # 화면 반영  #############################################################
            debug_image = cv.imshow('MediaPipe Holistic', debug_image)


            # 키처리 (ESC：종료) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                stop('out')
                time.sleep(0.3)
                goto_origin(0,100,0,100)
                break


    cap.release()
    cv.destroyAllWindows()

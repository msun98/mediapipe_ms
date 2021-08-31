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
# w_calibration, h_calibration = 1600, 900
mid_x, mid_y = w_calibration/2, h_calibration/2
neck_x,neck_y = 0,0 
point_flag = True
# point_flag = False
lock = threading.Lock()



def goto_human():
    global point_flag,predict_x,predict_y,cpx, cpy,neck_x,neck_y
    angle_of_x_old, angle_of_y_old = 0, 0
    pp, tp = 0, 0
    pan = 0
    predict_x_old = 0
    start = time.time()

    while True:
        lock.acquire()
        if point_flag:
            # print('working')

            if np.abs(pan-angle_of_x_old) > 500:
                moveTo(int(pan*100),int(tilt*100),0,0)

            else:
                # view2sphere(cpx, cpy, 0)
                # pan, tilt = camera2world()
                pan, tilt = np.rad2deg(calculate_alpha(neck_x,0)),np.rad2deg(calculate_beta(neck_y,0))
                # stop('right')
                # stop('up')
                # TO CALCULATE OF MOTOR SPEED
                end = 0.01
                # print(np.abs(pan-angle_of_x_old))
                angular_speed_x, angular_speed_y = round((np.abs(pan-angle_of_x_old)/end)*0.8,3),\
                                                   round((np.abs(tilt-angle_of_y_old)/end)*0.3,3) # 각속도 계산
                #
                pp, tp = int(0.825 * np.abs(angular_speed_x) + 0.127), int(0.825 * np.abs(angular_speed_y) + 0.127)


                if pan < 0:
                    print('\npan < 0\n')
                    if tilt > 0:
                        move_pan_tilt('right', 'up', int(pp/2), int(tp/3))
                    else:
                        move_pan_tilt('right', 'down', int(pp/2), int(tp/3))

                elif pan > 0:
                    print('\npan > 0\n')
                    if tilt > 0:
                        move_pan_tilt('left', 'up', int(pp/2), int(tp/3))
                    else:
                        move_pan_tilt('left', 'down', int(pp/2), int(tp/3))

                elif np.abs(pan- w_calibration/2)>10:
                    moveTo(int(pan*100),int(tilt*100),0,0)
                    print('\nstop\n')

                angle_of_x_old, angle_of_y_old = pan, tilt

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

    # 경계 사각형을 그릴 것인지 여부 (기본값 : 지정되지 않음)
    parser.add_argument('--use_brect', action='store_true')
    # parser.add_argument('--use_brect', action='store_false')

    # World 좌표를 matplotlib보기 ※ matplotlib를 이용하기 때문에 처리가 무거워집니다(기본값 : 지정되지 않음)
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args



if __name__ == '__main__':
    # main()
    # 인자분석 #################################################################
    args = get_args()
    cap = initialize() # 화면을 받아옴.
    on_screen_display()
    i = 1


    # initial_position = get_position()
    # if initial_position != (0, 0, 0):
    #     goto_origin(pp,ps,tp,ts)
    #     # time.sleep(0.3)

    # upper_body_only = args.upper_body_only
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

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
        goto_origin(pp,ps,tp,ts)
        # time.sleep(0.3)

    human = threading.Thread(target=goto_human)
    human.daemon = True  # 프로그램 종료시 즉시 종료.
    human.start()

    # FPS측정 모듈 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        if cap.isOpened():
            display_fps = cvFpsCalc.get()

            # 카메라 캡쳐 #####################################################
            ret, image = cap.read()
            if not ret:
                break

            if i == 1:
                goto_origin(pp,ps,tp,ts)
                time.sleep(0.3)
                i = 2

            image = calibration(image, w_calibration, h_calibration)
            debug_image = copy.deepcopy(image)

            # 감지 실시 #############################################################
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Face Mesh ###########################################################
            face_landmarks = results.face_landmarks
            # print(face_landmarks)
            if face_landmarks is not None:
                # 경계 사각형의 계산
                brect = calc_bounding_rect(debug_image, face_landmarks)
                # debug_image = cv2.rectangle(debug_image, int(brect[0]),int( brect[2]), (0,0,255), 2)
                # 얼굴 그림.
                debug_image = cv2.rectangle(debug_image, (brect[0], brect[1]), (brect[2], brect[3]), (0,255,0), 2)
                # 描画
                # debug_image = draw_face_landmarks(debug_image, face_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                # debug_image = cv2.rectangle(debug_image, (face_landmarks[0], startY), (endX, endY), (0,255,0), 2)

            # Pose ###############################################################
            pose_landmarks = results.pose_landmarks
            if pose_landmarks is not None:
                # 경계 사각형의 계산
                brect = calc_bounding_rect(debug_image, pose_landmarks)
                # 그리기 
                debug_image = draw_pose_landmarks(
                    debug_image,
                    pose_landmarks,
                    # upper_body_only,
                )
                pose = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = pose[0]
                right_sholder_x,right_sholder_y,left_sholder_x,left_sholder_y=pose[1],pose[2],pose[3],pose[4]

                # 목 위치 
                point_flag = True
                # print(point_flag)
                neck_x,neck_y = int((pose[1]+pose[3])/2) ,int((pose[2]+pose[4])/2) 
                cv.circle(debug_image, (neck_x, neck_y), 5, (0, 0, 255), 2)
                # point_flag = True

            
            # Hands ###############################################################
            left_hand_landmarks = results.left_hand_landmarks
            right_hand_landmarks = results.right_hand_landmarks
            # 왼손
            if left_hand_landmarks is not None:
                # 손바닥 중심 계산 
                cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
                # 경계사각형의 계산
                # brect = calc_bounding_rect(debug_image, left_hand_landmarks)
                # 그리기 
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    left_hand_landmarks,
                    # upper_body_only,
                    'R',
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
            # 오른손 
            if right_hand_landmarks is not None:
                # 손바닥 중심 계산 
                cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
                # 경계사각형의 계산
                # brect = calc_bounding_rect(debug_image, right_hand_landmarks)
                # 그리기 
                debug_image = draw_hands_landmarks(
                    debug_image,
                    cx,
                    cy,
                    right_hand_landmarks,
                    # upper_body_only,
                    'L',
                )
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            # 화면 반영  #############################################################
            cv.imshow('MediaPipe Holistic', debug_image)


            # 키처리 (ESC：종료) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                goto_origin(pp,ps,tp,ts)
                time.sleep(0.3)
                break

            # # 화면 반영  #############################################################
            # cv.imshow('MediaPipe Holistic', debug_image)

    cap.release()
    cv.destroyAllWindows()

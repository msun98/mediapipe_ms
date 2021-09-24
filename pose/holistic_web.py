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
pp, tp = 0, 0

dt = 1/20
sigmaX, sigmaA = 0.01, 0.015
cpx, cpy = 0, 0


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
kalman.measurementNoiseCov = np.array(R, np.float32) * sigmaX**2

neck_x, neck_y = w_calibration/2, h_calibration/2

kalman.statePre = np.array([[neck_x], [0], [0], [neck_y],[0],[0]], np.float32)
kalman.statePost = np.array([[neck_x], [0], [0], [neck_y],[0],[0]], np.float32)

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

    # upper_body_only = args.upper_body_only
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # 카메라 준비 ###############################################################
    # cap = cv.VideoCapture('vtest.avi')
    cap = cv.VideoCapture(0)

    # 모델로드 #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS측정 모듈 ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=30)

    while True:
        if cap.isOpened():
            display_fps = cvFpsCalc.get()

            # 카메라 캡쳐 #####################################################
            ret, image = cap.read()
            if not ret:
                break

            image = cv.flip(image, 1)
            debug_image = copy.deepcopy(image)

            start_camera = time.time()
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
                # pose = draw_bounding_rect(use_brect, debug_image, brect)
                pose = debug_image
                right_sholder_x,right_sholder_y,left_sholder_x,left_sholder_y = pose[1],pose[2],pose[3],pose[4]

                # 목 위치 
                # point_flag = True
                # print(point_flag)
                neck_x,neck_y = int((pose[1]+pose[3])/2) ,int((pose[2]+pose[4])/2) 
                point_flag = True
                end_time = time.time()-start_camera
                # print(end_time)
                debug_image = cv.circle(pose[0], (neck_x, neck_y), 5, (0, 0, 255), 2)
                # point_flag = True
            current_prediction = kalman.predict() # start kalman filter <predict>
            cpx, cpy = int(current_prediction[0]), int(current_prediction[3])
            if np.abs(cpy-neck_y) > 500:
                cpy = int(h_calibration/2)
            current_measurement = np.array([neck_x, neck_y], np.float32)
            kalman.correct(current_measurement)
            debug_image = cv2.circle(debug_image, (cpx, cpy), 8, (255, 0, 0), -1)

            
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
                
                if np.abs((debug_image[5][1]+debug_image[2][1])-(debug_image[3][1]+debug_image[4][1])) < 10:
                    # (검지 + 새끼) - (중지 + 약지)
                    print('손 접힘')

                    if debug_image[1][0]-debug_image[2][0] < 0:
                        print('왼쪽\n')

                    else:
                        if np.abs(debug_image[1][1]-cy) < 50:
                            print('손 다 접힘\n')
                        else:
                            print('오른쪽\n')

                else:
                    print('손펼침')
                debug_image = debug_image[0]

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
                debug_image = debug_image[0]

            cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            # 화면 반영  #############################################################
            cv.imshow('MediaPipe Holistic', debug_image)


            # 키처리 (ESC：종료) #################################################
            key = cv.waitKey(1)
            if key == 27:  # ESC
                time.sleep(1)
                break

    cap.release()
    cv.destroyAllWindows()

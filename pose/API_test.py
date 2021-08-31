# PTZ 쓰레드 사용하여 사람 따라가는 코드.

import argparse
import logging
import threading
import numpy as np
from calculate_rotation import *
from initial import *
from ptz_api import *
import time

sc = 0.8
sh,sw=int(sc*1080),int(sc*1920) # 화면크기
pp,ps,tp,ts=0,50,0,50
h_calibration,w_calibration = 833,1527
fps_time = 0


if __name__ == '__main__':

    vcap = initialize() # 화면을 받아옴.
    on_screen_display()
    time.sleep(0.3)

    initial_position = get_position()
    if initial_position != (0, 0, 0):
        goto_origin(pp,ps,tp,ts)
        time.sleep(0.5)
    pre_time = time.time()
    fps_data = []
    while True:
        if vcap.isOpened():
            #vcap.set(cv2.CAP_PROP_FPS, 20)
            ret_val, image = vcap.read()
            #vcap.set(cv2.CAP_PROP_FPS, 20)
            if not ret_val:
                break
            video = calibration(image, w_calibration, h_calibration)
            now_time = time.time()

            delta = now_time - pre_time
            fps = 1/delta
            if fps > 30:
                fps = 30
            fps_data.append(fps)
            cv2.putText(video, "FPS: %3.1f" % (fps), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow('tf-pose-estimation result', video)
            # out.write(video) # 영상 다운로드 하는 코드.
            # time.sleep(0.5)
            item = [5, 10, 5]
            for panpos in item:
                move('right',panpos) # 팬 각도가 음수일 때 오른쪽으로 회전.
                #time.sleep(0.1)
            pre_time = now_time

        k = key_event()

        if k == 27:
            goto_origin(pp,ps,tp,ts)
            break

    out.release()
    vcap.release()
    cv2.destroyAllWindows()


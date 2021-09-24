import cv2
import numpy as np

TITLE = "mouse tracking with acc"
frame = np.zeros((800,800,3),np.uint8)
numOfTail=100

def mousemove(event, x, y, s, p):
    global frame, current_measurement, current_prediction,cnt
    if cnt%4==0:
        current_measurement = np.array([[np.float32(x+10*np.random.randn())], [np.float32(y+10*np.random.randn())]])
    current_prediction = kalman.predict()
    cmx, cmy = int(current_measurement[0]), int(current_measurement[1])
    cpx, cpy = int(current_prediction[0]), int(current_prediction[3])
    frame = np.zeros((800,800,3),np.uint8)
    cv2.putText(frame, f"Measurement: ({cmx}, {cmy})",
                (30, 30), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (0, 255, 0),2, cv2.LINE_AA)
    cv2.putText(frame, f"Prediction: ({cpx}, {cpy})",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 70, 255),2, cv2.LINE_AA)
    cv2.putText(frame, f"True: ({x}, {y})",
                (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255, 255),2, cv2.LINE_AA)
    meas.append((cmx,cmy))
    preds.append((cpx,cpy))
    trues.append((x,y))
    for i,coord in enumerate(meas):
        cv2.circle(frame, coord, 3, (0, 150+i, 0), -1)      # current measured point
    for i,coord in enumerate(preds):
        cv2.circle(frame, coord, 3, (0, 0, 150+i), -1)      # current predicted point
    for i,coord in enumerate(trues):
        cv2.circle(frame, coord, 3, (150+i, 150+i,150+i), -1)      # true point
    if len(meas)>numOfTail:
        meas.pop(0)
    if len(preds)>numOfTail:
        preds.pop(0)
    if len(trues)>numOfTail:
        trues.pop(0)    
    kalman.correct(current_measurement)
    cnt+=1
sigmaX,sigmaY,sigmaA=0.1,0.1,0.015
dt=0.03
F=[ [1, dt, 0.5*dt**2, 0, 0, 0],
      [0, 1, dt, 0, 0, 0],
      [0, 0, 1, 0, 0, 0],
      [0, 0, 0, 1, dt, 0.5*dt**2],
      [0, 0, 0, 0, 1, dt],
      [0, 0, 0, 0, 0, 1]]
H=[ [1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0]]
Q=[ [dt**4/4, dt**3/2, dt**2/2, 0, 0, 0],
      [dt**3/2, dt**2, dt, 0, 0, 0],
      [dt**2/2, dt, 1, 0, 0, 0],
      [0, 0, 0, dt**4/4, dt**3/2, dt**2/2],
      [0, 0, 0, dt**3/2,dt**2, dt],
      [0, 0, 0, dt**2/2, dt, 1]]
R= [[1,0],[0,1]]
cv2.namedWindow(TITLE)
cv2.setMouseCallback(TITLE, mousemove)
meas=[]
preds=[]
trues=[]
kalman = cv2.KalmanFilter(6, 2)   #  state [ x, vx, ax, y, vy, ay]
kalman.transitionMatrix = np.array(F, np.float32)
kalman.measurementMatrix = np.array(H, np.float32) # System measurement matrix
kalman.processNoiseCov = np.array(Q, np.float32)*sigmaA**2 # System process noise covariance
kalman.measurementNoiseCov = np.array(R, np.float32) * sigmaX**1
cnt=0
while True:
    cv2.imshow(TITLE,frame)
    if cv2.waitKey(1) & 0xFF ==27:
        break
cv2.destroyAllWindows()

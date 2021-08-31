# 화각 및 구면 좌표값 계산
import numpy as np
from ptz_api import *
import sympy as sym

x_w, y_w, z_w = 0, 0, 0
h,w = 833,1527 # 캘리브레이션 이후의 값이므로 바꾸지 말 것.
horizon_aov_max = 56.4
horizon_aov_min = 3.1
vertical_aov_max = 33.5
vertical_aov_min = 1.7

def initialize_pt_variable():
    global x_w, y_w, z_w
    x_w=y_w=z_w=0

def calculate_alpha(u_p,zoom_position):
    # zoom_position = int(get_position()[2])
    new_u_p = u_p-(w / 2)
    th_w = float(horizon_aov_max-(horizon_aov_max-horizon_aov_min)/65535*zoom_position)
    alpha = th_w * new_u_p/w
    alpha = -alpha
    return np.deg2rad(alpha)


def calculate_beta(v_p,zoom_position):
    new_v_p=(h/2)-v_p
    # zoom_position = int(get_position()[2])
    pi_h = float(vertical_aov_max-(vertical_aov_max-vertical_aov_min)/65535*zoom_position)
    beta = pi_h * new_v_p / h
    return np.deg2rad(beta)


def view2sphere(u_p,v_p,zoom_position): # 화면 좌표 to 월드좌표
    global x_c, y_c, z_c
    
    alpha, beta = calculate_alpha(u_p,zoom_position), calculate_beta(v_p,zoom_position)
    print()
    print('alpha, beta :{:7.4f}, {:7.4f}'.format(np.rad2deg(alpha), np.rad2deg(beta)))
    cos_a, cos_b = np.cos(alpha), np.cos(beta)
    sin_a, sin_b = np.sin(alpha), np.sin(beta)


    # 카메라 좌표계에서 본 점의 좌표 
    pt_camera = np.array([[0],[0],[1]])
    
    R_y_alpha = np.array([[cos_a, 0, sin_a],
                         [     0, 1,     0],
                         [-sin_a, 0, cos_a]])

    R_x_beta = np.array([[1,     0 ,     0],
                         [0, cos_b,  sin_b],
                         [0, -sin_b, cos_b]])

    relative_pt_world = np.ravel(R_y_alpha @ R_x_beta @ pt_camera) # 고정 y축 회전 -> 회전 x축 회전.
    #print('relative point is ({:7.4f}, {:7.4f}, {:7.4f})'.format(*relative_pt_world))
    x_c = relative_pt_world[0]
    y_c = relative_pt_world[1]
    z_c = relative_pt_world[2]
    #print('카메라 좌표계에서의 (x, y, z) = ({:7.4f}, {:7.4f}, {:7.4f})'.format(x_c, y_c, z_c))
    return x_c, y_c, z_c


def camera2world(): # 월드좌표 to 구면좌표
    global PI,TH
    POS = get_position()

 # #----카메라 회전 각도를 입력받음.------##
    th_c = np.deg2rad(float(int(POS[0])/100)) # 한 번 움직일 때 2.7도 만큼 회전.(pan)
    pi_c = np.deg2rad(float(int(POS[1])/100)) # 한 번 움직일 때 3.xx도 만큼 회전.(tilt)

    cos_pi_c, sin_pi_c = np.cos(pi_c), np.sin(pi_c)
    cos_th_c, sin_th_c = np.cos(th_c), np.sin(th_c)

 # #----for 구면좌표계------##    
    Th_c = np.matrix([[  cos_th_c,  0, sin_th_c],
                      [         0,  1,  0      ],
                      [ -sin_th_c,  0, cos_th_c]])

    PI_c = np.matrix([[1,        0,         0],
                      [0, cos_pi_c,  sin_pi_c],
                      [0, -sin_pi_c, cos_pi_c]])

    p_w =  Th_c @ PI_c @ [[x_c], [y_c], [z_c]]

    x_w = p_w[0,0] # 1행 1열 
    y_w = p_w[1,0] # 1행 2열 
    z_w = p_w[2,0] # 1행 2열

    r = np.sqrt(x_w**2 + y_w**2 + z_w**2)
    squar = np.sqrt(r**2-y_w**2)
    TH = np.rad2deg(np.arctan2(x_w/squar,z_w/squar))
    if TH < 0:
        TH = 360+TH
    PI = np.rad2deg(np.arcsin(y_w/r))

    #print('World coordinate(PAN, TILT) : [{:0.5f} {:0.5f}]'.format(TH, PI))
    return TH,PI


def goto_want_point(): # 월드 좌표를 주면 그 곳으로 감.
    global X_W, Y_W, Z_W

    theta, pi = sym.Symbol('theta'), sym.Symbol('pi')

    X_W, Y_W, Z_W = input('x 좌표값 입력 : '),input('y 좌표값 입력 : '),input('z 좌표값 입력 : ')
    X_W, Y_W, Z_W = float(X_W), float(Y_W), float(Z_W)
    r = np.sqrt(X_W**2 + Y_W**2 + Z_W**2)
    print('거리값 : r',r)

    squar = np.sqrt(r**2-y_w**2)
    theta = np.rad2deg(np.arctan2(X_W/squar,Z_W/squar))
    if theta < 0:
        theta = 360+theta
    pi = np.rad2deg(np.arcsin(Y_W/r))
    print(theta,pi)
    
    return theta, pi
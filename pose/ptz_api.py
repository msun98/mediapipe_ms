# 움직임 제어함수, 좌표계 위치(move & stop : PTZ status, moveTo : Pan/Tilt control)
import requests
import re

#global_p, global_t = 0, 0
id_,pass_='admin','admin'
web_addr='192.168.0.9'
PTZ_head='http://'+web_addr+'/cgi-bin/control/'


def on_screen_display(): # 화면 내 글자 삭제
    url = PTZ_head + 'osd.cgi'\
    +'?id='+id_+'&passwd='+pass_+'&action='+'setosd'+'&texton=disable'+'&dateon=disable'+'&timeon=disable'
    response = requests.get(url, timeout=5)
    #print(url)

def get_frame_info():
    url = PTZ_head + 'capabilityvideo.cgi'\
    +'?id='+id_+'&passwd='+pass_+'&action='+'getFramerate'
    response = requests.get(url, timeout=2)
    print(response.text)


def get_video_info():
    url = PTZ_head+'control/videoset.cgi'\
    +'?id='+id_+'&passwd='+pass_+'&action='+'getvideo'
    response = requests.get(url, timeout=2)
    print(response.text)


def goto_origin(pp,ps,tp,ts): # 원점으로 가요 
    head=PTZ_head+'ptzf_status.cgi'
    head=head+'?id='+id_+'&passwd='+pass_
    head=head+'&action=goptzfpos'
    head=head+'&panpos='+str(pp)
    head=head+'&panspeed='+str(ps)
    head=head+'&tiltpos='+str(tp)
    head=head+'&tiltspeed='+str(ts)
    print(head)
    try:
        resp = requests.get(head,timeout=1)
    except Exception as ex:
        print(ex, 'Can not connect !!')


def move_pan_tilt(_pan,_tilt,pan_speed,tilt_speed): # moving same time pan, tilt
    
    if tilt_speed > 30:
        tilt_speed = 30

    if pan_speed > 30:
        pan_speed = 30

    if _pan == 'right' or _pan =='left':
        device = 'pan'
    # elif _pan == 'stop':
    #     device = 'stop'

    if _tilt == 'up' or _tilt =='down':
        device ='tilt'
    # elif _tilt == 'stop':
    #     device = 'stop'

    if device =='tilt' or  device == 'pan':
        head=PTZ_head+'pt_control.cgi'
        head=head+'?id='+id_+'&passwd='+pass_
        head=head+'&action=setptmove'
        
        if pan_speed == 0:
            head=head+'&pan='+ 'stop'
        else:
            head=head+'&pan='+_pan

        head=head+'&panspeed='+str(pan_speed)
        head=head+'&tilt='+_tilt
        if tilt_speed == 0:
            head=head+'&tilt='+ 'stop'

        else:
            head=head+'&tiltspeed='+str(tilt_speed)       
        #http://192.168.0.9/cgi-bin/control/pt_control.cgi?id=admin&passwd=admin&action=setptmove&pan=30&tilt=stop
        #(팬값만 돌고, 틸트는 멈춤.) 
    # if device=='pan':
    #     head=head+'&pan=stop'
    # elif device=='tilt' :
    #     head=head+'&tilt=stop'
    else:
        return
    print(head)
    try:
        #pass
        resp = requests.get(head,timeout=1)
    except Exception as ex:
        print('Can not connect !!')



def move(name,speed): # moving
    if name=='up' or name=='down':
        device='tilt'
    elif name=='in' or name=='out':
        device='zoom'
    else:
        device='pan'

    if device=='tilt' or  device=='pan':
        head=PTZ_head+'pt_control.cgi'
        head=head+'?id='+id_+'&passwd='+pass_
        head=head+'&action=setptmove'
    else:
        head=PTZ_head+'zf_control.cgi'
        head=head+'?id='+id_+'&passwd='+pass_
        head=head+'&action=setzfmove'           
    if device=='pan':
        head=head+'&pan='+name
        head=head+'&panspeed='+str(speed)
    elif device=='tilt' :
        head=head+'&tilt='+name
        head=head+'&tiltspeed='+str(speed)
    elif device=='zoom' :
        head=head+'&zoom='+name
        head=head+'&zoomspeed='+str(speed)           
    else:
        return
    print(head)
    try:
        #pass
        resp = requests.get(head,timeout=1)

        # print('출력중: ' + str(resp))
    except Exception as ex:
        print('Can not connect !!')


def stop(name): #stop
    if name=='up' or name=='down':
        device='tilt'
    elif name=='in' or name=='out':
        device='zoom'
    else:
        device='pan'
    if device=='tilt' or  device=='pan':
        head=PTZ_head+'pt_control.cgi'
        head=head+'?id='+id_+'&passwd='+pass_
        head=head+'&action=setptmove'
    else:
        head=PTZ_head+'zf_control.cgi'
        head=head+'?id='+id_+'&passwd='+pass_
        head=head+'&action=setzfmove'           
    if device=='pan':
        head=head+'&pan=stop'
    elif device=='tilt' :
        head=head+'&tilt=stop'
    elif device=='zoom' :
        head=head+'&zoom=stop'          
    else:
        return
    print(head)
    try:
        #pass
        resp = requests.get(head,timeout=1)
    except Exception as ex:
        print('Can not connect !!')


def moveTo(_pan, _tilt, pan_speed, tilt_speed):

    # if _pan > 36000 or _pan < 0:
    #     _pan %= 36000

    # if _tilt > 7000 :
    #     _tilt = 7000
    
    if _pan > 36000:
        _pan = _pan - 36000

    elif _pan < 0:
        _pan = 36000 - _pan 

    if _tilt > 7000 :
        _tilt = 7000

    elif _tilt < -1000:
        _tilt = -1000

    head=PTZ_head+'ptzf_status.cgi'
    head=head+'?id='+id_+'&passwd='+pass_
    head=head+'&action=goptzfpos' 
    head=head+'&panpos='+str(_pan)
    head=head+'&panspeed='+str(pan_speed)
    head=head+'&tiltpos='+str(_tilt)
    head=head+'&tiltspeed='+str(tilt_speed)
    
    # print(head)
    try:
        #pass
        resp = requests.get(head,timeout=2)
    except Exception as ex:
        print('Can not connect !!')


def get_position(): # 현재 ptz카메라의 위치를 알기 위해 사용.(각도로 나옴.)
    head = PTZ_head+'ptzf_status.cgi'
    head = head+'?id='+id_+'&passwd='+pass_
    head = head+'&action=getptzfpos'
    resp = requests.get(head)
    rs_code = resp.status_code
    if int(rs_code) == 200: # 현재 위치 불러오는 코드
        lis = resp.text
        numbers1 = re.findall("[a-z]+",lis) # 문자만 추출
        numbers2 = re.findall(".\d+",lis) # 숫자만 추출
        data = dict(zip(numbers1,numbers2)) # make list 2 dictionary
        panpos = data['panpos']
        tiltpos = data['tiltpos']
        zoompos = data['zoompos']

        return int(panpos), int(tiltpos), int(zoompos)
    else:
        print("fail")

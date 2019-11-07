import time
import socket

def start_Robot(brew_type):

    if brew_type==1:
        load="load /programs/coffee_demo/DEMO_ghost4_strongwater.urp\n"
    elif brew_type==2:
        load="load /programs/coffee_demo/DEMO_ghost4_slowwater.urp\n"
    else :
        return print("robot> error: brew_type")

   # HOST = "140.123.97.167"    # The remote host
    HOST = "192.168.1.2"    # The remote host

    PORT = 29999              # The same port as used by the server
    PORT2 = 30003
    robot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot.connect((HOST, PORT))

 #   robotScript = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   # robotScript.connect((HOST, PORT2))
    data = robot.recv(1024)
    print ("robot> Received:", repr(data))
    #dashbord
    poweron="power on\n"
    brakerelease="brake release\n"
    init="load installation /programs/u108_coffee_test/default.installation\n"
    play="play\n"
   #script 
    initMovej="  movej([-1.0262544790851038, -1.1924670378314417, 2.0154342651367188, -0.8343752066241663, -1.0188658873187464, -0.019378010426656544], a=0.5235987755982988, v=0.3490658503988659)\n"
    a=-0.50
    RGopen="RG2(40,40,0.0,True,False,False)\n"
    RGpick="RG2(0,40,0.0,True,False,False)\n"
    cmd="set_digital_out(1,True)\n"
    cmd2="movej([ "+ str(a) +", -2.350330184112267, -1.316631037266588, -2.2775736604458237, 3.3528323423665642, -1.2291967454894914], a=1.3962634015954636, v=1.0471975511965976)\n"
    RG2="RG2(0)\n"

    time.sleep(1)
    robot.send (load.encode("utf-8"))
    data = robot.recv(1024)
    print ("robot> Received:", repr(data))
    time.sleep(1)
    robot.send (poweron.encode("utf-8"))
    data = robot.recv(1024)
    print ("robot> Received:", repr(data))
    time.sleep(3)
    robot.send (brakerelease.encode("utf-8"))
    data = robot.recv(1024)
    print ("robot> Received:", repr(data))
    time.sleep(3)
    robot.send (play.encode("utf-8"))
    data = robot.recv(1024)
    print ("robot> Received:", repr(data))
# robot.send (brakerelease.encode("utf-8"))
# time.sleep(5)
    robot.close()
# str_command = "movel(p[%(x)s,%(y)s,%(z)s,%(rx)s,%(ry)s,%(rz)s],a=%(a)s,v=%(v)s,t=%(t)s)" %{'x':x,'y':y,'z':z,'rx':rx,'ry':ry,'rz':rz,'a':a,'v':v,'t':t}
# robot.send(str_command + "\n")
    return print("robot> start robot success!")
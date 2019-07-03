import time
import socket

def start_Robot(brew_type):

    if brew_type==1:
        load="load /programs/u108_coffee_test/finalcoffee19-05-03.urp"+"\n"
    else :
        return print("robot> error: brew_type")

    HOST = "140.123.97.167"    # The remote host
    PORT = 29999              # The same port as used by the server
    PORT2 = 30003
    robot = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robot.connect((HOST, PORT))

    robotScript = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    robotScript.connect((HOST, PORT2))
    #dashbord
    poweron="power on\n"
    brakerelease="brake release\n"
    init="load installation /programs/u108_coffee_test/default.installation\n"
    
   #script 
    initMovej="movej([-1.076204124943592, -1.2535856809951138, 2.1838314393273532, -0.887537832067828, -1.0542344275563273, 0.6796506164420769], a=0.5235987755982988, v=0.3490658503988659)\n"
    play="play\n"
    a=-0.50
    RGopen="RG2(40,40,0.0,True,False,False)\n"
    RGpick="RG2(0,40,0.0,True,False,False)\n"
    cmd="set_digital_out(1,True)\n"
    cmd2="movej([ "+ str(a) +", -2.350330184112267, -1.316631037266588, -2.2775736604458237, 3.3528323423665642, -1.2291967454894914], a=1.3962634015954636, v=1.0471975511965976)\n"
    RG2="RG2(0)\n"

    time.sleep(1)
    robot.send (load.encode("utf-8"))
    time.sleep(3)
    robot.send (poweron.encode("utf-8"))
    time.sleep(10)
    robot.send (brakerelease.encode("utf-8"))
    time.sleep(3)
    robot.send (play.encode("utf-8"))
    time.sleep(3)
# robot.send (brakerelease.encode("utf-8"))
# time.sleep(5)
    data = robot.recv(1024)
    robot.close()
    print ("robot> Received:", repr(data))
# str_command = "movel(p[%(x)s,%(y)s,%(z)s,%(rx)s,%(ry)s,%(rz)s],a=%(a)s,v=%(v)s,t=%(t)s)" %{'x':x,'y':y,'z':z,'rx':rx,'ry':ry,'rz':rz,'a':a,'v':v,'t':t}
# robot.send(str_command + "\n")
    return print("robot> start robot success!")
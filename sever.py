import socket

HOST='127.0.0.1'
PORT=8080

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(5)
c,addr=s.accept()
request=c.recv(1024)

print ('request is: ',request)
print ("\n--------------------------\n")
print ('Connected by', addr)
while True:
    request=c.recv(1024)
    print ('request is: ',request)
   
c.close()


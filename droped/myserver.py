import socket

HOST='127.0.0.1'
PORT=8000

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind((HOST,PORT))
s.listen(5)
c,addr=s.accept()
request=c.recv(1024)

print ('request is: ',request)
print ('Connected by', addr)
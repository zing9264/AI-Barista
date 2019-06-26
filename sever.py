from websocket_server import WebsocketServer
import time
islock= 0;

# Called for every client connecting (after handshake)
def new_client(client, server):
    global islock
    print("New client connected and was given id %d" % client['id'])
    if  islock==1:
        server.send_message_to_all("locking")
        return
    else :
        server.send_message_to_all("unlock")

# Called for every client disconnecting
def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
    global islock
    if  islock==1:
        print("locking proccess!")
        server.send_message_to_all("locking")
        return
    else :
        server.send_message_to_all("unlock")
    islock=1
    time.sleep(3);
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) : %s" % (client['id'], message))
    fp = open("filename.txt", "a")
    fp.write(message)
    fp.close()
    islock=0
 
    server.send_message_to_all("unlock")
    print("unlock proccess!")


PORT=8080
server = WebsocketServer(PORT)
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()
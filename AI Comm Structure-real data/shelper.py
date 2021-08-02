import socket

sckt = None
#conn = None
#addr = None

def send_data(sckt,content):
	sckt.send(content)

def read_data(sckt):
	buf = sckt.recv(1024)
	return buf

def close_socket(sckt):
	sckt.close()

def socket_setup(host, port):
	sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sckt.connect((host, port))
	return(sckt)


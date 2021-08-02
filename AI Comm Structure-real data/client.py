#!/usr/bin/env python3

import time
import random
from shelper import socket_setup
from shelper import close_socket
from shelper import read_data
from shelper import send_data
from checkPiece import Get_Move_Wrapper

from comm_functions import send_piece

HOST = "192.168.0.133"  # The server's hostname or IP address
PORT = 10236       # The port used by the server

#******************once game has started***********************************************
#get colour choice from the player set as this for now
#choose randonally for AI? 
rand = random.randint(0,1)
if rand == 0:
    mycolour = "Blac"
else:
    mycolour = "Whit"

#send colour choice 
s = socket_setup(HOST,PORT)
print("connected")
send_data(s, bytes(mycolour, "utf-8"))
print("Writing:" + mycolour)

#listen to find out what colour I am and if I  go first or second
data = read_data(s)
close_socket(s)
mesg = data.decode("utf-8")
print("Reading" + mesg)

#decode data so we know what to do
for i in range(0, len(mesg), 4):
    code = mesg[i:i+4]
    if(code == "0000"):
        # I go first
        turn = 0
    elif(code == "1111"):
        #I go second
        turn = 1
    elif(code == "Blac"):
        mycolour = "Blac"
        print("I am Black")
        col = 1

    elif(code == "Whit"):
        mycolour = "Whit"
        print("I am White")
        col = 2


#this is only needed when going second so that you are listening 
if(turn == 1):
    send_piece(HOST, PORT, "void",col)
    turn = 0
    

#remove when actual code can get location from user

while(1): #in actual implmentation 
    #play our piece
    #*************************************************************************
    #get piece move from the game board 
    move =  Get_Move_Wrapper(col) #return "x.y"  also could return "Skip" or "!sur"
    #**************************************************************************
    #send Piece move
    result = send_piece(HOST, PORT, move, col)
    if result == "!": #game over
        send_piece(HOST, PORT, "void",col)
        break
    elif result == "FAIL":
        print("Connection Lost")
        break

    result = send_piece(HOST, PORT, "void", col)
    if result == "!": #game over
        break
    elif result == "FAIL":
        print("Connection Lost")
        break



if result != "FAIL":
    winner = send_piece(HOST, PORT, "WINR",col)
    if winner == "B":
        print("Black Won!")
        #*******************************************************
        #LCD_Print("Game Over Black Won")
        #*******************************************************
    elif winner == "W":
        print("White Won!")
        #*******************************************************
        #LCD_Print("Game Over White Won")
        #*******************************************************
    else:
        print("error in final score")

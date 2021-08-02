import time

from shelper import socket_setup
from shelper import close_socket
from shelper import read_data
from shelper import send_data
from checkPiece import Add_Piece_Wrapper
from checkPiece import Remove_Piece_Wrapper


def send_piece(host, port, piece_added, colour):
    sckt = socket_setup(host, port)
    send_data(sckt, bytes(piece_added, "utf-8"))
    print("Writing:" +  bytes(piece_added, "utf-8").decode("utf-8"))

    data = read_data(sckt)
    mesg = data.decode("utf-8")
    print("Reading:" + mesg)
    close_socket(sckt)

    for i in range(0, len(mesg), 4):
        code = mesg[i:i+4]
        if(code[0] == "A"):
            x = int(code[1])
            y = int(code[3])
            #add a Piece to the board in the location of x.y the next three chars in string
            #***********************************************************************************
            Add_Piece_Wrapper(str(x) + "." + str(y), colour)           
             #************************************************************************************
            pass
        elif(code[0] == "R"):
            x = int(code[1])
            y = int(code[3])
            #Remove a Piece from the board in the location of x.y the next three chars in string
            #***********************************************************************************
            Remove_Piece_Wrapper(str(x) + "." + str(y))
            #************************************************************************************            
            pass
        elif(code == "Skip"):
            #opponent decided not to play their turn
            #*******************************************************
            #LCD_Print("Opponent Skipped their turn")
            print("Opponent Skiped")
            #*******************************************************
            pass
        elif(code[0] == "!"):
            #launch end game stage or
            #opponent surrendered 
            return("!")
        elif(code[0] == "B"):
            #last two digits are blacks score
            black_score =  code[-2:]
            print(black_score)
        elif(code[0] == "W"):
            #last two digits are blacks score
            white_score =  code[-2:]
            print(white_score)
            if(white_score < black_score):
                #black won 
                return "B"
            else:
                #white won
                return "W"
        elif(code == "FAIL"):
            return("FAIL")
    return()


#from __future__ import print_function
#initializing 3 different Lists

import random
import os
from keras.models import Model
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation



legal_moves_left = [1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,\
                    2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,\
                    3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,\
                    4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,\
                    5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,\
                    6.1,6.2,6.3,6.4,6.5,6.6,6.7,6.8,6.9,\
                    7.1,7.2,7.3,7.4,7.5,7.6,7.7,7.8,7.9,\
                    8.1,8.2,8.3,8.4,8.5,8.6,8.7,8.8,8.9,\
                    9.1,9.2,9.3,9.4,9.5,9.6,9.7,9.8,9.9]

all_game_states = []

#black 1 white 2 in array
#          1 2 3 4 5 6 7 8 9   
row0 =  [3,3,3,3,3,3,3,3,3,3,3] #
row1 =  [3,0,0,0,0,0,0,0,0,0,3] #9
row2 =  [3,0,0,0,0,0,0,0,0,0,3] #18
row3 =  [3,0,0,0,0,0,0,0,0,0,3] #27
row4 =  [3,0,0,0,0,0,0,0,0,0,3] #36
row5 =  [3,0,0,0,0,0,0,0,0,0,3] #45
row6 =  [3,0,0,0,0,0,0,0,0,0,3] #54
row7 =  [3,0,0,0,0,0,0,0,0,0,3] #63
row8 =  [3,0,0,0,0,0,0,0,0,0,3] #72
row9 =  [3,0,0,0,0,0,0,0,0,0,3] #81
row10 = [3,3,3,3,3,3,3,3,3,3,3]

col = [row0, row1, row2, row3, row4, row5, row6, row7, row8, row9, row10]

test_board = np.array([[[
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0,
]]]).reshape(1, 9, 9, 1)

#Globals
checked1 = []
colour7 = 0
points = 0
myColour = 0
notMyColour = 0
move_probs = []

#used to create a model in order to generate probalities 
def model_creation1():
    global test_board
    global move_probs

    np.random.seed(123)
    X = np.load('features-40k.npy')
    Y = np.load('labels-40k.npy')

    samples = X.shape[0]
    size = 9
    input_shape = (size, size, 1)
    X = X.reshape(samples, size, size, 1)

    train_samples = int(0.9 * samples)
    X_train, X_test = X[:train_samples], X[train_samples:]
    Y_train, Y_test = Y[:train_samples], Y[train_samples:]
    # end::mcts_go_cnn_preprocessing[]

    # tag::mcts_go_cnn_model[]
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3),activation='relu',padding='same',input_shape=input_shape))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(48, (3, 3),padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(size * size, activation='softmax'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    # end::mcts_go_cnn_model[]

    checkpoint_path = "training_1/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # end::mcts_go_cnn_eval[]

    move_probs = model.predict(test_board)[0]
    i = 0
    for row in range(9):
        row_formatted = []
        for col in range(9):
            row_formatted.append('{:.3f}'.format(move_probs[i]))
            i += 1
        print(' '.join(row_formatted))

def model_creation2():
    global test_board
    global move_probs

    # np.random.seed(123)
    # X = np.load('features4.npy')
    # Y = np.load('labels4.npy')

    # samples = X.shape[0]
    size = 9
    input_shape = (size, size, 1)
    # X = X.reshape(samples, size, size, 1)

    # train_samples = int(0.9 * samples)
    # X_train, X_test = X[:train_samples], X[train_samples:]
    # Y_train, Y_test = Y[:train_samples], Y[train_samples:]
    # end::mcts_go_cnn_preprocessing[]

    # tag::mcts_go_cnn_model[]
    model = Sequential()
    model.add(ZeroPadding2D(padding=3,input_shape=input_shape))
    model.add(Conv2D(48,(3,3)))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))

    model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(size * size, activation='softmax'))
    #model.add(Activation('relu'))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    # end::mcts_go_cnn_model[]

    checkpoint_path = "training_2/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Loads the weights
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    # score = model.evaluate(X_test, Y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # end::mcts_go_cnn_eval[]

    move_probs = model.predict(test_board)[0]
    i = 0
    for row in range(9):
        row_formatted = []
        for col in range(9):
            row_formatted.append('{:.3f}'.format(move_probs[i]))
            i += 1
        print(' '.join(row_formatted))



def get_max():
    global move_probs
    # Get the maximum element from a Numpy array
    # maxElement = np.amax(move_probs)
    # print('Max element from Numpy Array : ', maxElement)
    # Get the indices of maximum element in numpy array
    result = np.where(move_probs == np.amax(move_probs))
    # print('Returned tuple of arrays :', result)
    # print('List of Indices of maximum element :', result[0])

    return result

#calculates the final score
def finalScore(myList=[], *args):

    check = []
    check1 = []
    empty_size = 0
    whiteScore = 0
    blackScore = 0
    global myColour
    global notMyColour


    global colour7
    global points

    for x in range(11):
        for y in range(11):
            if col[y][x] == 0:
                if (y*9-9+x) not in check1:
                    pieceCovered(x,y,check,myList)
                    myColour = 0
                    notMyColour = 0

                    if colour7 == 1:
                        blackScore = blackScore + points
                    elif colour7 == 2:
                        whiteScore = whiteScore + points
                    colour7 = 0
                    points = 0
                    check1.extend(check)
                    check.clear()

            elif col[y][x] == 1:
                blackScore += 1
                check1.append((y)*9-9+(x))

            elif col[y][x] == 2:
                whiteScore += 1
                check1.append(y*9-9+x)

            
   # whiteScore = whiteScore + 7.5
    print("Black:" + str(blackScore) + " White:"+ str(whiteScore))
    return ([blackScore, whiteScore])

#used in calculating the final score
def pieceCovered(x,y,check,myList = [], *args):
    
    global colour7 
    global points
    global myColour
    global notMyColour

    colour = myList[y][x]
 
    check.append(y*9-9+x)
    points = points + 1

    count = [0,1,2,3]

    xdir = [1,-1,0,0]
    ydir = [0,0,1,-1]
    
    
    for i in count:
        
        if myList[y+ydir[i]][x+xdir[i]] == 1:
            if myColour == 0:
                myColour = 1
                notMyColour = 2
                colour7 = 1
            elif myColour == 1:
                pass
            elif myColour == 2:
                points = 0    
                return(int(0))
        elif myList[y+ydir[i]][x+xdir[i]] == 2:
            if myColour == 0:
                myColour = 2
                notMyColour = 1
                colour7 = 2
            elif myColour == 2:
                pass
            elif myColour == 1: 
                points = 0  
                return(int(0))

        elif myList[y+ydir[i]][x+xdir[i]] == 0:
           # if it is the same
            if ((y+ydir[i])*9-9+(x+xdir[i])) not in check:
                # if we have not checked it yet
                if pieceCovered(x+xdir[i],y+ydir[i],check,myList) == 0:
                    points = 0
                    return(int(0))

# colour is the colour you're trying to capture, xloc, yloc of added piece
def pieceCaptured(x,y,checked,myList = [], *args):
    
    colour = myList[y][x]
 
    checked.append(y*9-9+x)
    
    count = [0,1,2,3]

    xdir = [1,-1,0,0]
    ydir = [0,0,1,-1]
    
    
    for i in count:
        
        if myList[y+ydir[i]][x+xdir[i]] == 0:
            checked.clear()
            return("empty")

        elif myList[y+ydir[i]][x+xdir[i]] == colour:
           # if it is the same
            if ((y+ydir[i])*9-9+(x+xdir[i])) not in checked:
                # if we have not checked it yet
                if pieceCaptured(x+xdir[i],y+ydir[i],checked,myList) == "empty":
                    checked.clear()
                    return("empty")

    return(checked)
        
def add_piece_to_testboard(x, y, colour):
    global test_board

    if colour == 2:
        colour = -1
    test_board[0][x -1][y -1][0] = colour

def remove_piece_from_testboard(x, y):
    global test_board

    test_board[0][x -1][y -1][0] = 0

def addPiece(location, colour ,myList = [], *args):
    global legal_moves_left
    global all_game_states
    loc_float= float(location) #make a float
    try:
        legal_moves_left.remove(loc_float) #remove the location from the legal moves list
    except:
        pass
    loc_x = int(loc_float)
    loc_y = int((loc_float - loc_x) * 10.1)

    myList[loc_y][loc_x] = colour
    add_piece_to_testboard(loc_x,loc_y,colour)


    all_game_states.append(board_to_string(myList))

    #print(myList)

def removePiece(location ,myList = [], *args):
    global legal_moves_left
    loc_float = float(location)
    legal_moves_left.append(loc_float)

    loc_x = int(loc_float)
    loc_y1 = ((loc_float - loc_x) * 10.1)
    loc_y = int(loc_y1)

    myList[loc_y][loc_x] = 0
    remove_piece_from_testboard(loc_x, loc_y)
   # print(myList[loc_y][loc_x])

def checkPeice(loc,checked,myList = [], *args):

    loc_float = float(loc)
    x = int(loc_float)
    yf = ((loc_float-x)*10.1)
    y = int(yf)

    count = [0,1,2,3]
    PiecestobeRemoved = ""
    #arrays to add to x and y direction
    xdir = [1,-1,0,0]
    ydir = [0,0,1,-1]
    
    colour = myList[y][x]

    for i in count:

        if myList[y+ydir[i]][x+xdir[i]] == colour:
            pass
        elif (myList[y+ydir[i]][x+xdir[i]] == 0) or (myList[y+ydir[i]][x+xdir[i]] == 3):
            pass
        else:
            #check if the pieces need to be removed sourounding this piece
            piecesNeedRemoved = pieceCaptured(x+xdir[i],y+ydir[i], checked, myList)
            if piecesNeedRemoved == (None or 'empty'):
                pass
            else:
                for j in piecesNeedRemoved:
                    #if rgw do need to be removed get into proper format and then remove piece
                     yloc = int((j-1)/9+1)
                     xloc = (int(((j-1)/9+1-yloc)*10))+1
                     formatPiece = str(xloc) + "." + str(yloc)
                     removePiece(formatPiece, myList)
                     formatPiece = "R" + formatPiece
                     PiecestobeRemoved = PiecestobeRemoved + formatPiece
    
    #check if the piece that was added needs to be removed
    thisPieceRemoved = pieceCaptured(x,y,checked,myList)
    if thisPieceRemoved == (None or 'empty'):
        pass
    else:
        for j in thisPieceRemoved:
            yloc = int((j-1)/9+1)
            xloc = (int(((j-1)/9+1-yloc)*10))+1
            formatPiece = str(xloc) + "." + str(yloc)
            removePiece(formatPiece, myList)
            formatPiece = "R" +formatPiece
            PiecestobeRemoved = PiecestobeRemoved + formatPiece


    return(PiecestobeRemoved)

def is_eye(x,y,myList = [], *args):
    colour = myList[y][x]
    count = [0,1,2,3]
    xdir = [1,-1,0,0]
    ydir = [0,0,1,-1]

    for i in count:
        if ((myList[y+ydir[i]][x+xdir[i]] == colour) or (myList[y+ydir[i]][x+xdir[i]] == 3)):
            pass
        else:
            return("noteye")
    return("eye")

def board_to_string(myList = [], *args):
    
    a = ""

    for i in range(11):
        for j in range(11):
            a += str(myList[i][j])
    return(a)

def AI_select_move(MyColour1,myList = [], *args):
    #randomally select location 
    global all_game_states
    global legal_moves_left
    check = []
    h = []
    while(1):
        rand = random.randint(0,len(legal_moves_left)-1) #get a random number to get legal piece to play

        loc_float = legal_moves_left[rand] #element at random number

        print("try:"+ str(loc_float))

        #convert to x and y values 
        loc_x = int(loc_float)
        loc_y = int((loc_float - loc_x) * 10.1)

        #add Piece to game board
        addPiece(loc_float,MyColour1,myList)
        #take off the addition from the list from addPiece called in this function
        all_game_states.pop()

        #check if there are any other pieces there
            #only empty spaces should be in legal_moves_left
        #check if it results in self capture
        if pieceCaptured(loc_x,loc_y,h,col) != "empty":
            #means that it is self capture have to pick new place
            pass
        elif is_eye(loc_x,loc_y,col) == "eye":
            #means that it is an eye have to pick new place
            pass
        elif board_to_string(myList) in all_game_states:
            #means this board state has already been made have to pick new piece

            pass
        else:
            #all is good
            all_game_states.append(board_to_string(myList)) #need to add back to game states if this is our move
            break

        #piece cannot be selected
        removePiece(loc_float,myList)
        check.append(loc_float)
        legal_moves_left.remove(loc_float) #remove the location from the legal moves list so we don't check it again

        if len(legal_moves_left) == 0:
            for i in check: #need to add each element back one at a time
                legal_moves_left.append(i)
            return("Skip")

    print("Worked")
    return(str(loc_float))
    
def Get_Move_Wrapper(Colour):
    global col
    value = AI_select_move2(Colour, col)
    return(value)

def Add_Piece_Wrapper(loc, Colour):
    global col
    addPiece(loc, Colour, col)

def Remove_Piece_Wrapper(loc):
    global col
    removePiece(loc,col)
    
def AI_select_move2(MyColour1,myList = [], *args):
    #randomally select location 
    global all_game_states
    global legal_moves_left
    global move_probs
    check = []
    h = []
    model_creation2()
    while(1):

        while(1):
            max = get_max()
            max = max[0]
            print(max)
            if (move_probs == 0).all() == True:
                return("Skip")
            #convert max to x & y

            loc_x = int(max % 9 + 1)
            loc_y = int((max - (max%9))/ 9 + 1)

            loc_float = loc_x + loc_y/10
        
            if myList[loc_y][loc_x] == 0:
                break
            else:
                move_probs[max] = 0

        #add Piece to game board
        addPiece(loc_float,MyColour1,myList)
        #take off the addition from the list from addPiece called in this function
        all_game_states.pop()

        #check if it results in self capture
        if pieceCaptured(loc_x,loc_y,h,col) != "empty":
            #means that it is self capture have to pick new place
            pass
        elif is_eye(loc_x,loc_y,col) == "eye":
            #means that it is an eye have to pick new place
            pass
        elif board_to_string(myList) in all_game_states:
            #means this board state has already been made have to pick new piece
            pass
        else:
            #all is good
            all_game_states.append(board_to_string(myList)) #need to add back to game states if this is our move
            break

        #piece cannot be selected
        removePiece(loc_float,myList)
        check.append(loc_float)
        #legal_moves_left.remove(loc_float) #remove the location from the legal moves list so we don't check it again
        move_probs[max] = 0

    return(str(loc_float))
   

# add_piece_to_testboard(9,8, 1)
# add_piece_to_testboard(9,9, 2)
# add_piece_to_testboard(5,5, 1)
# add_piece_to_testboard(5,6, 2)
# model_creation()
# get_max()


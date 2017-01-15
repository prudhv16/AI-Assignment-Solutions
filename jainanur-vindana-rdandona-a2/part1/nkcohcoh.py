'''
**************************************************************
Author :- Prudhvi Indana
Date :- 10/09/2016
Objective :- Generate next possible best move in n-k-koh-koh
using alpha beta pruning and mini max algorithm and a
scoring mechanism based on number of similar adjacent elements
adding depth and a heuristic function to make the program
search only untill a limited depth based on time.
**************************************************************
'''

import copy
import sys


def successor(board, n, currentPlayer):
    '''
    This function is used to generate next possible successor boards based on current board
    '''
    validMoves = [[i, j] for i in range(n) for j in range(n) if board[i][j] == '.']
    successorStates = []
    for i, j in validMoves:
        tempboard = copy.deepcopy(board)
        tempboard[i][j] = currentPlayer
        successorStates.append(tempboard)
    return successorStates


def heuristicscore(board, block, k):
    '''
    Calculates score of the given board and block based on
    number of blocks of size k available for black - number of blocks of size k available for white
    in case of terminal state it retuns 10**k if current player is black
    and if its terminal state and current player is white it returns -10**k
    
    '''
    blocklen = len(block)
    blackcount = 0;
    whitecount = 0
    if blocklen < k:
        return 0
    tempboard = []
    for i, j in block:
        tempboard.append(board[i][j])
    for i in range(blocklen - k + 1):
        tempblock = tempboard[i:k + i]
        tempwhitecount = tempblock.count("w");
        tempblackcount = tempblock.count("b")
        if tempwhitecount == 0:
            blackcount += 1
        if tempblackcount == 0:
            whitecount += 1
        if tempblackcount == k:
            return 10**k
        if tempwhitecount == k:
            return -10**k
    # print blackcount,whitecount
    return blackcount - whitecount

def checkVertical(board, position, n, k):
    '''
    for a given position generate all possible positions in the same column.
    '''
    checkList = [[i, position[1]] for i in range(n)]
    #print checkList
    # print checkmatch(board,checkList,k)
    return heuristicscore(board, checkList, k)


def checkHorizontal(board, position, n, k):
    '''
    for a given position generate all possible positions in the same row.
    '''
    checkList = [[position[0], j] for j in range(n)]
    # print checkList
    # return checkmatch(board,checkList,k)
    return heuristicscore(board, checkList, k)


def checkMajorDiagonal(board, position, n, k):
    '''
    for a given position get the positions of all the elements in its major diagonal.
    '''
    checkList = [[position[0] - i, position[1] - i] for i in range(n, 0, -1) if
                 position[0] - i >= 0 and position[1] - i >= 0] \
                + [[position[0] + i, position[1] + i] for i in range(n) if position[0] + i < n and position[1] + i < n]
    #print checkList
    # return checkmatch(board,checkList,k)
    return heuristicscore(board, checkList, k)


def checkMinorDiagonal(board, position, n, k):
    '''
    for a given position get the position of all the elements in its minor diagonal.
    '''
    # tempboard = [i[::-1] for i in board]'''
    #print position
    
    checkList = [[position[0] - i, position[1] + i] for i in range(n, 0, -1) if
                 position[0] - i >= 0 and position[1] + i < n] \
                + [[position[0] + i, position[1] - i] for i in range(n) if position[0] + i < n and position[1] - i >= 0]
    position = [position[0], n - position[1] - 1]
    #print checkList
    # return checkmatch(board,checkList,k)
    return heuristicscore(board, checkList, k)


def score(board, n, k):
    '''
    calculate score of the board by checking
    all rows of length greater than k
    all columns of length greater than k
    all major diagonals of length greater than k
    all minor diagonals of length greater than k
    '''
    scorevalue = 0
    #print "cal score"
    for i in range(n):
        scorevalue += checkVertical(board, [i, i], n, k)
        scorevalue += checkHorizontal(board, [i, i], n, k)
    scorevalue += checkMajorDiagonal(board, [0, 0], n, k)
    scorevalue += checkMinorDiagonal(board, [n-1, n-1], n, k)
    scorevalue += checkMinorDiagonal(board, [0, 0], n, k)
    for i in range(1, n):
        scorevalue += checkMajorDiagonal(board, [0, i], n, k)
        scorevalue += checkMajorDiagonal(board, [i, 0], n, k)
        scorevalue += checkMinorDiagonal(board, [n-1, n-i-1], n, k)
        #scorevalue += checkMinorDiagonal(board, [0, i], n, k)
        if i != n-1:
            scorevalue += checkMinorDiagonal(board, [i, 0], n, k)
    return scorevalue


def removeduplicates(states):
    '''
    when given a list of all possible states, this function returns a
    list removing all duplicate states by comparing it with its images on all 4 axises
    '''
    uniquestates = []
    # print len(states)

    for state in states:
        count = 0
        # nextstates = [state,state[::-1],[i[::-1] for i in state],[i[::-1] for i in state][::-1]]
        nextstates = [state, state[::-1], [i[::-1] for i in state]]
        nextstates.append(nextstates[2][::-1])
        # print nextstates
        for elem in nextstates:
            if elem in uniquestates:
                break
            else:
                count += 1
        if count == 4:
            uniquestates.append(nextstates[0])
    return uniquestates


def alphabetapruning(board, a, b, maximising, n, k, currentplayer, depth):
    '''
    search based onalpha beta pruning based on based on given board.
    '''
    beststate = None
    scorevalue = score(board, n, k)
    # print board
    # print scorevalue
    #if depth == 0 or not isinstance(scorevalue, int):
    if any([depth == 0,scorevalue >= 10**k-5,scorevalue <= -10**k-5]):
        #print scorevalue,depth,board
        return scorevalue
    nextplayer = 'b' if currentplayer == 'w' else 'w'
    # print nextplayer
    succStates = successor(board, n, currentplayer)
    succStates = removeduplicates(succStates)
    nextmaximising = copy.deepcopy(maximising);
    nexta = copy.deepcopy(a);
    nextb = copy.deepcopy(b)
    if maximising == True:
        for validboard in succStates:
            tempa = a
            a = max(a, alphabetapruning(validboard, nexta, nextb, not nextmaximising, n, k, nextplayer, depth - 1))
            if tempa != a:
                beststate = validboard
            if a >= b:
                break
        return a
    else:
        for validboard in succStates:
            tempb = b
            b = min(b, alphabetapruning(validboard, nexta, nextb, not nextmaximising, n, k, nextplayer, depth - 1))
            if tempb != b:
                beststate = validboard
            if a >= b:
                break
        return b

def alphabetapruningwithreturn(board, a, b, maximising, n, k, currentplayer, depth):
    '''
    search based on alpha beta pruning based on based on given board and return the best next possible of the board.
    '''
    beststate = None
    scorevalue = score(board, n, k)
    # print board
    # print scorevalue
    #if depth == 0 or not isinstance(scorevalue, int):
    #if any([depth == 0,scorevalue >= 10**k,scorevalue <= -10**k]):
    #    print board,scorevalue
    #    return scorevalue
    nextplayer = 'b' if currentplayer == 'w' else 'w'
    # print nextplayer
    succStates = successor(board, n, currentplayer)
    succStates = removeduplicates(succStates)
    nextmaximising = copy.deepcopy(maximising);
    nexta = copy.deepcopy(a);
    nextb = copy.deepcopy(b)
    if maximising == True:
        for validboard in succStates:
            tempa = a
            a = max(a, alphabetapruning(validboard, nexta, nextb, not nextmaximising, n, k, nextplayer, depth - 1))
            if tempa != a:
                beststate = validboard
            if a >= b:
                break
        return a, beststate
    else:
        for validboard in succStates:
            tempb = b
            b = min(b, alphabetapruning(validboard, nexta, nextb, not nextmaximising, n, k, nextplayer, depth - 1))
            if tempb != b:
                beststate = validboard
            if a >= b:
                break
        return b, beststate

if __name__ == "__main__":
    #board = "wbwbbww.."
    #n = 3
    #k = 3
    n,k,board,timelimit = int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],int(sys.argv[4])
    maxdepth = board.count('.')
    board = [list(board[i * n:i * n + n]) for i in range(n)]
    #print board
    #print "\n".join("".join(board[i]) for i in range(n))
    # print checkVertical(board,[0,0],n,k)
    # print checkHorizontal(board,[0,0],n,k)
    # print checkMajorDiagonal(board,[1,1],n,k)
    #print checkMinorDiagonal(board, [1, 1], n, k)
    whitecount = sum([item.count('w') for item in board])
    blackcount = sum([item.count('b') for item in board])
    currentplayer = "w" if blackcount >= whitecount else "b"
    # print currentplayer
    maximising = True if currentplayer == "w" else False
    a = -float("inf");
    b = float("inf")
    # print -1>b
    #print currentplayer
    #succStates = successor(board, n, currentplayer) 
    #print succStates
    #print removeduplicates(succStates)

    #print maxdepth
    depth = 1
    #depth = 7
    #print maximising
    #print currentplayer
    prevstatescore = None
    prevstate = None
    while depth <= maxdepth:
        currentresult =  alphabetapruningwithreturn(board,a,b,maximising,n,k,currentplayer,depth)
        #print currentresult
        #print currentresult
        if prevstate == None:
            prevstate = currentresult[1]
            prevstatescore = score(prevstate,n,k)
        currstate = currentresult[1]
        currscore = score(currstate,n,k)
        if currentplayer == "w" and currscore <= -10**k:
            currstate = prevstate
        elif currentplayer == "b" and currscore >= +10**k:
            currstate = prevstate
        else:
            prevstate = currstate
            prevstatescore = currscore
            #print currstate
        #currscore = score(prevresult[1],n,k)
        #tempboard = prunresult[1]
        #tempscore = score(tempboard,n,k)
        #print tempscore
        #if tempscore >= 10**k or tempscore <= -10**k:
        #    print 
        #else:
        print "".join("".join(currstate[i]) for i in range(n))
        depth += 1

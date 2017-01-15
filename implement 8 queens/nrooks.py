# nrooks.py : Solve the N-Rooks problem!
# D. Crandall, August 2016
#
# The N-rooks problem is: Given an empty NxN chessboard, place N rooks on the board so that no rooks
# can take any other, i.e. such that no two rooks share the same row or column.

# This is N, the size of the board.
from collections import deque
import numpy as np
N=8

# Count # of pieces in given row
def count_on_row(board, row):
    return sum( board[row] ) 

# Count # of pieces in given column
def count_on_col(board, col):
    return sum( [ row[col] for row in board ] ) 
    #return sum( board[:][col])

# Count total # of pieces on board
def count_pieces(board):
    return sum([ sum(row) for row in board ] )

# Return a string with the board rendered in a human-friendly format
def printable_board(board):
    return "\n".join([ " ".join([ "Q" if col else "_" for col in row ]) for row in board])

# Add a piece to the board at the given position, and return a new board (doesn't change original)
def add_piece(board, row, col):
    return board[0:row] + [board[row][0:col] + [1,] + board[row][col+1:]] + board[row+1:]

# Get list of successors of given board state
def successors(board):
    #temp_board = [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]
    #print count_pieces(temp_board)
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) ]

# Get list of successors of given board state
def successors2(board):
    board_count = count_pieces(board) 
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) 
if count_pieces(add_piece(board, r, c)) == board_count + 1 and board_count + 1 <= N ]

# Get list of successors of given board state
def successors3(board):
    board_count = count_pieces(board)
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) 
if count_pieces(add_piece(board, r, c)) == board_count + 1 and board_count + 1<= N and 
count_on_row(board,r) == 0 and count_on_col(board,c) == 0 ]

def successors3Nqueens(board):
    board_count = count_pieces(board)
    return [ add_piece(board, r, c) for r in range(0, N) for c in range(0,N) 
if count_pieces(add_piece(board, r, c)) == board_count + 1 and board_count + 1<= N and 
count_on_row(board,r) == 0 and count_on_col(board,c) == 0 and check_diag(board,r, c)]

# check if board is a goal state
def is_goal(board):
    return count_pieces(board) == N and \
        all( [ count_on_row(board, r) <= 1 for r in range(0, N) ] ) and \
        all( [ count_on_col(board, c) <= 1 for c in range(0, N) ] )


#Check diagonals to make sure no two queens are in same diagonal
def check_diag(board,r,c):
    board = np.array(board)
    return primdiagchecker(board,r,c) and primdiagchecker(board[:,::-1],r,N-c-1)

#Slices the board and sums diagonal elements and checks if there are any existing diagonal elements.
def primdiagchecker(board,r,c):
    #for i in board:
    #    print i
    if r >= c:
        d = r - c
        temp_board = board[d:,:N-d]
    #    print temp_board
        return np.trace(temp_board) == 0
    else :
        d = c - r
        temp_board = board[:N-d,d:]
    #    print temp_board
        return np.trace(temp_board) == 0

# Solve n-rooks!
def solve(initial_board):
    fringe = deque()
    fringe.append(initial_board)
    #fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors3( fringe.pop() ):
	    if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# Solve n-rooks!
def solveNqueens(initial_board):
    fringe = deque()
    fringe.append(initial_board)
    #fringe = [initial_board]
    while len(fringe) > 0:
        for s in successors3Nqueens( fringe.pop() ): 
	    if is_goal(s):
                return(s)
            fringe.append(s)
    return False

# The board is stored as a list-of-lists. Each inner list is a row of the board.
# A zero in a given square indicates no piece, and a 1 indicates a piece.
initial_board = [[0]*N]*N;
print "Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for nrook solution...\n"
solution = solve(initial_board)
print printable_board(solution) if solution else "Sorry, no solution found. :("
print "Starting from initial board:\n" + printable_board(initial_board) + "\n\nLooking for nqueen solution...\n"
solution = solveNqueens(initial_board)
print printable_board(solution) if solution else "Sorry, no solution found. :("

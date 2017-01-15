# Tetris backend v0.2
# D. Crandall, Sept 2016
#
# DON'T MODIFY THIS FILE! Or else we might not be able to grade your submission properly.
#


import sys, time, random, threading, thread

class EndOfGame(Exception):
  def __init__(self,s) :
    self.str = s
  
  def __str__(self):
    return self.str

class TetrisGame:

  PIECES = [ [ "xxxx" ], [ "xx ", " xx" ], [ "xx", "xx" ], [ "xxx", "  x"], [ "xxx", " x " ] ]
  BOARD_HEIGHT = 20
  BOARD_WIDTH = 10

  # initialize empty board. State is a pair with the board in first element and score in the second.
  def __init__(self):
    self.state = ([ " " * TetrisGame.BOARD_WIDTH ] * TetrisGame.BOARD_HEIGHT, 0)
    self.piece_dist = [ [i,] * random.randint(0, 10) for i in range(0, len(TetrisGame.PIECES) ) ]
    self.piece_dist = [ i for m in self.piece_dist for i in m ]
    self.next_piece = None
    self.new_piece()
    
  # rotate a given piece by a given angle
  @staticmethod
  def rotate_piece(piece, rotation):
    rotated_90 = [ "".join([ str[i] for str in piece[::-1] ]) for i in range(0, len(piece[0])) ]
    return { 0: piece, 90: rotated_90, 180: [ str[::-1] for str in piece[::-1] ], 270: [ str[::-1] for str in rotated_90[::-1] ] }[rotation]

  def random_piece(self):
    return TetrisGame.rotate_piece(TetrisGame.PIECES[ random.choice(self.piece_dist) ], random.randrange(0, 360, 90) ) 

  # print out current state to the screen
  @staticmethod
  def print_state((board, score)):
    print "\n" * 3 + ("Score: %d \n" % score) + "|\n".join(board) + "|\n" + "-" * TetrisGame.BOARD_WIDTH

  # return true if placing a piece at the given row and column would overwrite an existing piece
  @staticmethod
  def check_collision((board, score), piece, row, col):
      return col+len(piece[0]) > TetrisGame.BOARD_WIDTH or row+len(piece) > TetrisGame.BOARD_HEIGHT \
          or any( [ any( [ (c != " " and board[i_r+row][col+i_c] != " ") for (i_c, c) in enumerate(r) ] ) for (i_r, r) in enumerate(piece) ] )
    
  # take "union" of two strings, e.g. compare each character of two strings and return non-space one if it exists
  @staticmethod
  def combine(str1, str2):
      return "".join([ c if c != " " else str2[i] for (i, c) in enumerate(str1) ] )

  # place a piece on the board at the given row and column, and returns new (board, score) pair
  @staticmethod
  def place_piece((board, score), piece, row, col):
    return (board[0:row] + \
              [ (board[i+row][0:col] + TetrisGame.combine(r, board[i+row][col:col+len(r)]) + board[i+row][col+len(r):] ) for (i, r) in enumerate(piece) ] + \
              board[row+len(piece):], score)

  # remove any "full" rows from board, and increase score accordingly
  @staticmethod
  def remove_complete_lines((board, score)):
    complete = [ i for (i, s) in enumerate(board) if s.count(' ') == 0 ]
    return ( [(" " * TetrisGame.BOARD_WIDTH),] * len(complete) + [ s for s in board if s.count(' ') > 0 ], score + len(complete) )

  # move piece left or right, if possible
  def move(self, col_offset, new_piece):
    new_col = max(0, min(TetrisGame.BOARD_WIDTH - len(self.piece[0]), self.col + col_offset))
    (self.piece, self.col) = (new_piece, new_col) if not TetrisGame.check_collision(self.state, new_piece, self.row, new_col) else (self.piece, self.col)

  def finish(self):
      self.state = TetrisGame.remove_complete_lines( TetrisGame.place_piece(self.state, self.piece, self.row, self.col) )      
      self.new_piece()

  def new_piece(self):
      # generate a new random piece to fall at a random position
      self.piece = self.next_piece if self.next_piece != None else self.random_piece()
      self.next_piece = self.random_piece()
      self.col = random.randrange(0, TetrisGame.BOARD_WIDTH - len(self.piece[0]))
      self.row = 0

      # check if this immediately generates a collision, which means we lost!
      if(TetrisGame.check_collision(self.state, self.piece, self.row, self.col)):
        raise EndOfGame("Game over! Final score: " + str( self.state[1]))

  def print_board(self, clear_screen):
    if clear_screen: print "\n"*80
    print "Next piece:\n" + "\n".join(self.next_piece)
    TetrisGame.print_state(TetrisGame.place_piece(self.state, self.piece, self.row, self.col))

  ######
  # These are the "public methods" that your code might want to call!
  #
 
  # move piece left, if possible, else do nothing
  def left(self):
    self.move(-1, self.piece)

  # move piece right, if possible, else do nothing
  def right(self):
    self.move(1, self.piece)

  # rotate piece one position if possible, else do nothing
  def rotate(self):
    self.move(0, TetrisGame.rotate_piece(self.piece, 90))

  # make piece go all the way down until it hits a collision
  def down(self):
    while not TetrisGame.check_collision(self.state, self.piece, self.row+1, self.col):
      self.row += 1
    self.finish()

  # return current state of board
  def get_board(self):
    return self.state[0]

  # return current score
  def get_score(eslf):
    return self.score

  # return currently-falling piece, and its current row and column on the boar
  def get_piece(self):
    return (self.piece, self.row, self.col)

  # return next piece 
  def get_next_piece(self):
    return self.next_piece
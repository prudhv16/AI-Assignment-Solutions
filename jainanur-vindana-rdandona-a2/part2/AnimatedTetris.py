# "Fancier" animated interactive version of tetris. v0.2
# D. Crandall, Sept 2016
#
# DON'T MODIFY THIS FILE! Or else we might not be able to grade your submission properly.
#

from TetrisGame import *

class AnimatedTetris(TetrisGame):

  def __init__(self):
      TetrisGame.__init__(self)

  # This thread just repeated displays the current game board to the screen.
  def display_thread(self):
    while 1:
      self.print_board(True)
      time.sleep(0.1)

  # This thread is in charge of making the piece fall over time.
  def gravity_thread(self):
    while True:
      while 1:
        time.sleep(0.5)
        self.row = self.row+1
        if(TetrisGame.check_collision(self.state, self.piece, self.row+1, self.col)): break

      # place new piece in final resting spot 
      self.finish()

  # This thread just starts things up
  def start_game(self, player):
    t2 = threading.Thread(target=self.gravity_thread)
    t2.setDaemon(True)
    t3 = threading.Thread(target=self.display_thread)
    t3.setDaemon(True)
    t2.start()
    t3.start()

    player.control_game(self)
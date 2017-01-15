import termios, fcntl, sys, os

#####
# Following function is based on:
# http://stackoverflow.com/questions/13207678/whats-the-simplest-way-of-detecting-keyboard-input-in-python-from-the-terminal/31736883
def get_char_keyboard():
  fd = sys.stdin.fileno()

  oldterm = termios.tcgetattr(fd)
  newattr = termios.tcgetattr(fd)
  newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
  termios.tcsetattr(fd, termios.TCSANOW, newattr)

  c = None
  try:
    c = sys.stdin.read(1)
  except IOError: pass

  finally:
    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)

  return c
#####
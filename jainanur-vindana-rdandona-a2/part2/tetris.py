# Simple tetris program! v0.2
# D. Crandall, Sept 2016
#
# Code Description:
# The evaluation function computes a score by considering the current state of the board, the current piece (all
# positions) and the upcoming piece (all positions).
#
# The following is computed for each possible placement of the current piece on the board and the upcoming piece:-
# Aggregate Height: Sum of the heights of all the columns of the board
# Holes: Sum of the total number of holes (an empty space with at least one tile in the same column above it)
# Bumpiness: the variation of the column heights
# Complete Lines
#
# These 4 parameters are used to compute the score as follows:
# a x(Aggregate Height) + b x(Complete Lines) + c x(Holes) + d x(Bumpiness)
#
# where a = -0.510066, b = 0.760666, c = -0.35663, d = -0.184483
#
# a,c,d are weights to minimize aggregate height, holes and bumpiness (hence negative). b is to maximize Complete Lines
#
# The state (current board, current piece and upcoming piece) with the best score is used for the placement of the
# current piece
#


from AnimatedTetris import *
from SimpleTetris import *
from TestBoard import *
from kbinput import *
import time, sys


class HumanPlayer:

    def get_moves(self, tetris):
        print "Type a sequence of moves using: \n  b for move left \n  m for move right \n  n for rotation\n" \
              "Then press enter. E.g.: bbbnn\n"
        moves = raw_input()
        return moves

    def control_game(self, tetris):
        while 1:
            c = get_char_keyboard()
            commands = {"b": tetris.left, "n": tetris.rotate, "m": tetris.right, " ": tetris.down}
            commands[c]()

            if tetris.col == 0 and len(sys.argv) == 4:
                if sys.argv[3] == "adversarial":
                    """worst_next_piece = self.advesrse()"""
                    """tetris.next_piece = worst_next_piece"""


    def initializeMyBoard(self, tetris):
        x_tetris = TestBoard()
        x_tetris.piece = tetris.piece
        x_tetris.col = tetris.col
        x_tetris.row = tetris.row
        x_tetris.state = tetris.state
        x_tetris.next_piece = tetris.get_next_piece()
        return x_tetris


    def advesrse(self):
        worst_piece = ""

        x_tetris = self.initializeMyBoard(tetris)
        new_player = ComputerPlayer()
        x_move_for_current_piece = new_player.get_moves_adversial(x_tetris)
        commands = {"b": x_tetris.left, "n": x_tetris.rotate, "m": x_tetris.right, " ": x_tetris.down}

        for c in x_move_for_current_piece:
            commands[c]()

        x_tetris.piece = x_tetris.get_next_piece()
        x_move_for_current_piece = new_player.get_moves_adversial(x_tetris)
        for c in x_move_for_current_piece:
            commands[c]()

        worst_piece = new_player.adversarial(x_tetris)
        return worst_piece


#####
# This is the part you'll want to modify!
# Replace our super simple algorithm with something better
#
class ComputerPlayer:

    def possibleRoatations(self, x_tetris):
        # Function to find all possible unique pieces(considering roatation) that can be achived
        # from the current piece/next piece
        possiblePieces = []
        possibleNextPieces = []
        i = 0
        my_piece = x_tetris.get_piece()[0]
        my_next_piece = x_tetris.get_next_piece()
        while i < 4:
            if my_piece not in possiblePieces:
                possiblePieces.append(my_piece)
            i += 1
            my_piece = x_tetris.rotate_piece(my_piece, 90)

            if my_next_piece not in possibleNextPieces:
                possibleNextPieces.append(my_next_piece)
            my_next_piece = x_tetris.rotate_piece(my_next_piece, 90)
        return [possiblePieces, possibleNextPieces]


    def possibleMoves(self, my_pieces, x_tetris):
        # Function to find possible moves of current piece and next piece. For next piece the columns is considered as 0
        x_moves = []
        x_next_piece_moves = []

        # Possible rotations of current piece
        x_rotation = ""
        for x_piece in my_pieces[0]:
            x_move = ""
            i = x_tetris.get_piece()[2]  # current column
            while i <= 10 - len(x_tetris.get_piece()[0][0]):
                x_moves.append(x_rotation + x_move)
                x_move += "m"
                i += 1
            i = x_tetris.get_piece()[2] - 1  # current column - 1
            x_move = "b"
            while i >= 0:
                x_moves.append(x_rotation + x_move)
                x_move += "b"
                i -= 1
            x_tetris.rotate()
            x_rotation += "n"

        # Possible moves of next_piece considering piece at col = 0
        x_rotation = ""
        for x_piece in my_pieces[1]:
            x_move = ""
            i = 0  # considering next piece at 0 column
            x_next_piece = x_tetris.get_next_piece()
            while i <= 10 - len(x_next_piece[0][0]):
                x_next_piece_moves.append(x_rotation + x_move)
                x_move += "m"
                i += 1
            x_next_piece = x_tetris.rotate_piece(x_piece, 90)
            x_rotation += "n"
        return [x_moves, x_next_piece_moves]


    def checkMoves(self, x_moves, tetris):
        # x_move_Result = []                  # list of list to store results of move i
        best_move_list = ["", 0]
        first_move_not_checked = True
        for x_move in x_moves[0]:
            for x_next_move in x_moves[1]:
                x_tetris = self.initializeMyBoard(tetris)

                # Commands
                COMMANDS = {"b": x_tetris.left, "n": x_tetris.rotate, "m": x_tetris.right}
                for c in x_move:
                    COMMANDS[c]()

                x_current_score = x_tetris.state[1]

                # Moving the current piece to bottom
                while not x_tetris.check_collision(x_tetris.state, x_tetris.piece, x_tetris.row + 1, x_tetris.col):
                    x_tetris.row += 1
                x_tetris.state = x_tetris.remove_complete_lines(x_tetris.place_piece(x_tetris.state, x_tetris.piece,
                                                                                     x_tetris.row, x_tetris.col))

                # Bringing in the next piece
                x_tetris.piece = x_tetris.get_next_piece()
                x_tetris.row = 0
                x_tetris.col = 0
                for c in x_next_move:
                    COMMANDS[c]()

                # Moving the next_piece to bottom
                while not x_tetris.check_collision(x_tetris.state, x_tetris.piece, x_tetris.row + 1, x_tetris.col):
                    x_tetris.row += 1

                # Deleting completed lines
                x_tetris.state = x_tetris.remove_complete_lines(x_tetris.place_piece(x_tetris.state, x_tetris.piece,
                                                                                     x_tetris.row, x_tetris.col))
                x_new_score = x_tetris.state[1]
                x_completed_lines = x_new_score - x_current_score

                result_list = self.checkMoveResult(x_tetris.get_board(), x_completed_lines)

                x_current_score = self.calcScore(result_list)
                # x_move_Result.append(x_current_score)

                if first_move_not_checked:
                    first_move_not_checked = False
                    best_move_list = [x_move, x_next_move, x_current_score]

                if x_current_score > best_move_list[2]:
                    best_move_list[0] = x_move
                    best_move_list[1] = x_next_move
                    best_move_list[2] = x_current_score

        # print "BM:" + str(best_move_list)
        return best_move_list


    def calcScore(self, result_list):
        # The weights are based on result published in
        # https://www2.informatik.uni-erlangen.de/publication/download/mic.pdf
        # weights = [-62709, -30271, 0, 48621, -35395, -12, -43810, 0, 0, -4041, -44262, -5832]

        # the weights are based on research given in
        # https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/

        return -0.5100*float(result_list[0]) + 0.760666*float(result_list[3]) \
               -0.35663*float(result_list[2]) - 0.184483*float(result_list[1])

    """else:
            result = 0.0
            i = 0
            while i < len(list):
                result += float(weights[i]*list[i])
                i += 1
            return result"""            # For 2nd evaluation function

    def checkMoveResult(self, test_board, x_completed_lines):
        x_each_column_height = self.getEachColumnHeight(test_board)
        x_aggregate_height = 0                  # Sum of all heights

        x_bumpiness = 0                         # Sum of differences between te heights of adjacent columns heights
        x_last_height = x_each_column_height[0]

        """x_max_height = 0
        x_min_height = x_each_column_height[0]"""           # For 2nd eval function include this
        for x_height in x_each_column_height:
            x_aggregate_height += x_height
            x_bumpiness += abs(x_height - x_last_height)
            x_last_height = x_height

            # For 2nd evaluation function include this in for loop
            """if x_max_height < x_height:
                x_max_height = x_height
            if x_min_height > x_height:
                x_min_height = x_height"""


        x_current_holes = self.getBoardHoles(test_board, x_each_column_height)
        return [x_aggregate_height, x_bumpiness, x_current_holes, x_completed_lines]

        # Below result used with 2nd evaluation function
        # x_altitude_difference = x_max_height - x_min_height
        # x_current_holes_wells = self.getBoardHoles_better(test_board, x_each_column_height)
        # x_current_holes = x_current_holes_wells[0]
        # x_current_connected_holes = x_current_holes_wells[1]
        # x_number_of_blocks = x_current_holes_wells[2]
        # x_weighted_blocks = x_current_holes_wells[3]

        # x_wells = self.getWells(x_each_column_height)
        # x_maximum_well_depth = x_wells[1]
        # x_sum_of_wells = x_wells[0]
        # solution = 1
        # x_landing_height = 0

        # return [x_max_height, x_current_holes, x_current_connected_holes, x_completed_lines, x_altitude_difference,
        # x_maximum_well_depth, x_sum_of_wells, x_landing_height, x_number_of_blocks, x_weighted_blocks]


    def getEachColumnHeight(self, test_board):
        result = []
        j = 0
        while j < 10:
            result.append(0)
            i = 0
            height = 0
            while i < 20:
                if test_board[i][j] == 'x':
                    result[j] = 20 - i
                    break
                i += 1
            j += 1
        return result


    def getBoardHoles(self, test_board, x_height_of_each_column):
        i = 19
        holes = 0
        while i > 0:
            number_of_spaces_in_row = 0
            j = 0
            while j < 10:
                if test_board[i][j] == ' ':
                    if (20 - i) < x_height_of_each_column[j]:
                        number_of_spaces_in_row += 1
                        holes += 1
                j += 1
                if number_of_spaces_in_row == 10:
                    break
            i -= 1
        return holes

    # Used for second eval function. Returns number of blocks by counting x, no of holes, no of vertical holes(
    # continuous vertical holes, weighted number of blocks(where block at height n has weight n))
    def getBoardHoles_better(self, test_board, x_height_of_each_column):
        i = 19
        holes = 0
        number_of_connected_holes = 0
        number_of_blocks = 0
        number_of_weighted_blocks = 0
        while i > 0:
            number_of_spaces_in_row = 0
            j = 0
            while j < 10:
                if test_board[i][j] == 'x':
                    number_of_blocks += 1
                    number_of_weighted_blocks += 20 - i

                if test_board[i][j] == ' ':
                    if (20 - i) < x_height_of_each_column[j]:
                        number_of_spaces_in_row += 1
                        holes += 1

                        if test_board[i-1][j] == 'x':
                            number_of_connected_holes += 1
                j += 1
                if number_of_spaces_in_row == 10:
                    break
            i -= 1
        return [holes, number_of_connected_holes, number_of_blocks, number_of_weighted_blocks]

    # Used in 2nd eval function. Returns number of wells and max height of a well
    def getWells(self, x_height_each_column):
        number_of_wells = 0
        maximum_well_depth = 0

        if x_height_each_column[0] < x_height_each_column[1]:
            number_of_wells += 1
            maximum_well_depth = x_height_each_column[1] - x_height_each_column[0]
        if x_height_each_column[9] < x_height_each_column[8]:
            number_of_wells += 1
            well_depth = x_height_each_column[8] - x_height_each_column[9]
            if maximum_well_depth < well_depth:
                maximum_well_depth = well_depth

        j = 1
        while j < 9:
            if x_height_each_column[j] < x_height_each_column[j-1] and \
                            x_height_each_column[j] < x_height_each_column[j+1]:
                number_of_wells += 1
                well_depth = x_height_each_column[j-1] - x_height_each_column[j]
                if well_depth < x_height_each_column[j+1] - x_height_each_column[j]:
                    well_depth = x_height_each_column[j + 1] - x_height_each_column[j]
                if maximum_well_depth < well_depth:
                    maximum_well_depth = well_depth
            j += 1
        return [number_of_wells, maximum_well_depth]


    # Makes a copy of current tetris board and returns it
    def initializeMyBoard(self, tetris):
        x_tetris = TestBoard()
        x_tetris.piece = tetris.piece
        x_tetris.col = tetris.col
        x_tetris.row = tetris.row
        x_tetris.state = tetris.state
        x_tetris.next_piece = tetris.get_next_piece()
        return x_tetris

    def adversarial(self, board):
        PIECES = [["xx ", " xx"], ["xxxx"], ["xx", "xx"], ["xxx", "  x"], ["xxx", " x "]]
        x_best_result_for_current_piece = []
        for x_piece in PIECES:
            x_tetris = self.initializeMyBoard(tetris)  # Makes copy of current board
            x_tetris.piece = x_piece
            x_tetris.next_piece = ""
            x_tetris.col = 0
            x_tetris.row = 0
            x_possible_pieces = self.possibleRoatations(x_tetris)  # Gets different possible rotations of piece/next piece
            x_possibleMoves = self.possibleMoves(x_possible_pieces, x_tetris)  # All possible moves for current + next piece
            x_best_move = self.checkMoves(x_possibleMoves,
                                          tetris)  # Best move with best move for next tile and its score
            x_best_result_for_current_piece.append(x_best_move[2])

        i = 0
        worst_next_piece_score = x_best_result_for_current_piece[0]
        worst_next_piece = PIECES[0]
        while i < 5:
            if worst_next_piece_score > x_best_result_for_current_piece[i]:
                worst_next_piece_score = x_best_result_for_current_piece[i]
                worst_next_piece = PIECES[i]
            i += 1
        return worst_next_piece


    # This function should generate a series of commands to move the piece into the "optimal"
    # position. The commands are a string of letters, where b and m represent left and right, respectively,
    # and n rotates. tetris is an object that lets you inspect the board, e.g.:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.

    def get_moves(self, tetris):
        x_tetris = self.initializeMyBoard(tetris)               # Makes copy of current board
        x_possible_pieces = self.possibleRoatations(x_tetris)   # Gets different possible rotations of piece/next piece
        x_possibleMoves = self.possibleMoves(x_possible_pieces, x_tetris)  # All possible moves for current + next piece
        x_best_move = self.checkMoves(x_possibleMoves, tetris)  # Best move with best move for next tile and its score
        return x_best_move[0]

    # For adversial
    def get_moves_adversial(self, tetris):
        x_tetris = self.initializeMyBoard(tetris)  # Makes copy of current board
        x_possible_pieces = self.possibleRoatations(x_tetris)  # Gets different possible rotations of piece/next piece
        x_possibleMoves = self.possibleMoves(x_possible_pieces, x_tetris)  # All possible moves for current + next piece
        x_best_move = self.checkMoves_adversial(x_possibleMoves,
                                                tetris)  # Best move with best move for next tile and its score
        return x_best_move[0]

    def checkMoves_adversial(self, x_moves, tetris):
        # x_move_Result = []                  # list of list to store results of move i
        best_move_list = ["", 0]
        first_move_not_checked = True
        for x_move in x_moves[0]:

            x_tetris = self.initializeMyBoard(tetris)

            # Commands
            COMMANDS = {"b": x_tetris.left, "n": x_tetris.rotate, "m": x_tetris.right}
            for c in x_move:
                COMMANDS[c]()

            x_current_score = x_tetris.state[1]

            # Moving the current piece to bottom
            while not x_tetris.check_collision(x_tetris.state, x_tetris.piece, x_tetris.row + 1, x_tetris.col):
                x_tetris.row += 1
            x_tetris.state = x_tetris.remove_complete_lines(x_tetris.place_piece(x_tetris.state, x_tetris.piece,
                                                                                 x_tetris.row, x_tetris.col))
            # Deleting completed lines
            x_tetris.state = x_tetris.remove_complete_lines(x_tetris.place_piece(x_tetris.state, x_tetris.piece,
                                                                                 x_tetris.row, x_tetris.col))
            x_new_score = x_tetris.state[1]
            x_completed_lines = x_new_score - x_current_score

            result_list = self.checkMoveResult(x_tetris.get_board(), x_completed_lines)

            x_current_score = self.calcScore(result_list)
            # x_move_Result.append(x_current_score)

            if first_move_not_checked:
                first_move_not_checked = False
                best_move_list = [x_move, "", x_current_score]

            if x_current_score > best_move_list[2]:
                best_move_list[0] = x_move
                best_move_list[1] = ""
                best_move_list[2] = x_current_score

        # print "BM:" + str(best_move_list)
        return best_move_list

    # This is the version that's used by the animated version. This is really similar to get_moves,
    # except that it runs as a separate thread and you should access various methods and data in
    # the "tetris" object to control the movement. In particular:
    #   - tetris.col, tetris.row have the current column and row of the upper-left corner of the
    #     falling piece
    #   - tetris.get_piece() is the current piece, tetris.get_next_piece() is the next piece after that
    #   - tetris.left(), tetris.right(), tetris.down(), and tetris.rotate() can be called to actually
    #     issue game commands
    #   - tetris.get_board() returns the current state of the board, as a list of strings.
    #
    def control_game(self, tetris):
        COMMANDS = {"b": tetris.left, "n": tetris.rotate, "m": tetris.right}
        # another super simple algorithm: just move piece to the least-full column
        while 1:
            x_moves = self.get_moves(tetris)
            for c in x_moves:
                COMMANDS[c]()

            tetris.down()


###################
# main program

(player_opt, interface_opt) = sys.argv[1:3]

try:
    if player_opt == "human":
        player = HumanPlayer()
    elif player_opt == "computer":
        player = ComputerPlayer()
    else:
        print "unknown player!"

    if interface_opt == "simple":
        tetris = SimpleTetris()
    elif interface_opt == "animated":
        tetris = AnimatedTetris()
    else:
        print "unknown interface!"

    if len(sys.argv) == 4:
        if sys.argv[3] == "adversarial" and player_opt == "human":
            tetris.piece = TetrisGame.PIECES[1]
            tetris.next_piece = tetris.PIECES[1]

    tetris.start_game(player)

except EndOfGame as s:
    print "\n\n\n", s

# Reference:
# https://codemyroad.wordpress.com/2013/04/14/tetris-ai-the-near-perfect-player/
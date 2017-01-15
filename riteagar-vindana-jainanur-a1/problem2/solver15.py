import math
import copy
import time
import sys
# 15 puzzle problem has many solutions and is in fact one of the challenging problems in AI.
# First I used Manhattan distance which seemed a good idea at first but was in vain because even though the computation
# time was less and the results were really fast for some puzzle configurations it took days to solve some of the configurations like
#  [[1,2,3,4],[9,10,11,12],[5,6,8,7],[13,15,14,0]]. The reason was because the distance is not effective enough as we have only
# one tile as the heuristic state and of the billions of state it just explores. Secondly I used linear conflict where there was a little
# improvement but not significant enough. Thirdly, I decided to divide the 15 puzzle into two disjoint patterns and calculate minimum moves for
# those puzzles and adding them to find the heuristics. Now, to calculate the minimum moves for those puzzles is not that easy but then I came across a
# concept called retrograde analysis where I use inverse BFS. For each disjoint pattern I used retrograde analysis to calculate the minimum moves and
# thus went down the tree of states. For example, If I divide the patterns into [1,2,3,4,5,6,7,8] and [9,10,11,12,13,14,15] then for the state say
# [[5,0,0,0], [0,3,0,1],[0,6,0,7],[2,0,4,8]], I take the minimum tile, 1 in this case and move it to the goal state. When I come across a tile from
# all the paths I remove that tile in all possible directions and increase the move and continue for 1 till it found its position. Now in a same level,
# 1 can find its position multiple ways and since I need to keep track of the minimum moves I take all the states of that level where 1 is in goal position.
#  I pass those states to the fringe to calculate for 2 now and this goes on till 8. For second pattern I do the same but instead of finding goal position
# for 1 I find goal position for 15, from reverse, as it will not collide for the rest. I do get the minimum moves for this and a much better heuristics,
# as it turns out configurations which can take days for Manhattan distance can be reduced to hours([[1,2,3,4],[9,10,11,12],[5,6,8,7],[13,15,14,0]]).
# Problems with the approach:
# It consumes a lot of memory to keep track of each disjoint pattern, even though temporarily. It takes time even for simple configurations,
# like for some cases if a configuration takes 25 seconds for Manhattan distance, it will take 600 seconds or more for my heuristic.
# If I want my heuristic to be consistent and admissible it may take up a lot of time, in which case it is going to find the optimal path.

class boards(object):
    def __init__(self, id, element, depth, visited):
        self.id = id
        self.element = element
        self.depth = depth
        self.visited = visited

class tile_node(object):
    def __init__(self, id, parent, depth):
        self.id = id
        self.parent = parent
        self.depth = depth

def move_up_board(tile12):
    tile1 = copy.deepcopy(tile12)
    row, column = position(0, tile1)
    if row == 0:
        x = tile1[row][column]
        tile1[row][column] = tile1[row+3][column]
        tile1 [row+3][column] = x
    else:
        x = tile1[row][column]
        tile1[row][column] = tile1[row-1][column]
        tile1[row-1][column] = x
    return tile1

def move_down_board(tile12):
    tile1 = copy.deepcopy(tile12)
    row, column = position(0, tile1)
    if row == 3:
        x = tile1[row][column]
        tile1[row][column] = tile1[row-3][column]
        tile1 [row-3][column] = x
    else:
        x = tile1[row][column]
        tile1[row][column] = tile1[row+1][column]
        tile1 [row+1][column] = x
    return tile1

def move_left_board(tile12):
    tile1 = copy.deepcopy(tile12)
    row, column = position(0, tile1)
    if column == 0:
        x = tile1[row][column]
        tile1[row][column] = tile1[row][column+3]
        tile1 [row][column+3] = x
    else:
        x = tile1[row][column]
        tile1[row][column] = tile1[row][column-1]
        tile1 [row][column-1] = x
    return tile1

def move_right_board(tile12):
    tile1 = copy.deepcopy(tile12)
    row, column = position(0, tile1)
    if column == 3:
        x = tile1[row][column]
        tile1[row][column] = tile1[row][column-3]
        tile1 [row][column-3] = x
    else:
        x = tile1[row][column]
        tile1[row][column] = tile1[row][column+1]
        tile1 [row][column+1] = x
    return tile1

def move_up(tile, board1):
    board = copy.deepcopy(board1)
    path_element = 0
    row, column = position(tile, board)
    if row == 0 and board[3][column] == 0:
        x = board[row][column]
        board[row][column] = board[row+3][column]
        board [row+3][column] = x
    elif row !=0 and board[row-1][column] == 0:
        x = board[row][column]
        board[row][column] = board[row-1][column]
        board[row-1][column] = x
    elif row == 0 and board[3][column] !=0:
        path_element = board[3][column]
    elif row !=0 and board[row-1][column] != 0:
        path_element = board[row-1][column]
    return board, path_element

def move_down(tile, board1):
    board = copy.deepcopy(board1)
    path_element = 0
    row, column = position(tile, board)
    if row == 3 and board[0][column] == 0:
        x = board[row][column]
        board[row][column] = board[row-3][column]
        board [row-3][column] = x
    elif row !=3 and board[row+1][column] == 0:
        x = board[row][column]
        board[row][column] = board[row+1][column]
        board[row+1][column] = x
    elif row == 3 and board[0][column] !=0:
        path_element = board[row-3][column]
    elif row !=3 and board[row+1][column] != 0:
        path_element = board[row+1][column]
    return board, path_element

def move_left(tile, board1):
    board = copy.deepcopy(board1)
    path_element = 0
    row, column = position(tile, board)
    if column == 0 and board[row][3] == 0:
        x = board[row][column]
        board[row][column] = board[row][column+3]
        board[row][column+3] = x
    elif column!=0 and board[row][column-1] == 0:
        x = board[row][column]
        board[row][column] = board[row][column-1]
        board[row][column-1] = x
    elif column == 0 and board[row][column+3] !=0:
        path_element = board[row][column+3]
    elif column !=0 and board[row][column-1] != 0:
        path_element = board[row][column-1]
    return board, path_element

def move_right(tile, board1):
    board = copy.deepcopy(board1)
    path_element = 0
    row, column = position(tile, board)
    if column == 3 and board[row][0] == 0:
        x = board[row][column]
        board[row][column] = board[row][column-3]
        board[row][column-3] = x
    elif column !=3 and board[row][column+1] == 0:
        x = board[row][column]
        board[row][column] = board[row][column+1]
        board[row][column+1] = x
    elif column == 3 and board[row][0] !=0:
        path_element = board[row][0]
    elif column !=3 and board[row][column+1] != 0:
        path_element = board[row][column+1]
    return board, path_element

#Expmnd the tiles only when certain conditions are satisfied like for each pattern I check the 1st tile to be placed first

def expand_next_element(tile, board1):
    l = []
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    board = copy.deepcopy(board1)
    a = move_up(tile, board)
    if a[0] != board:
        l.append(a[0])
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    b = move_down(tile, board)
    if b[0] != board:
        l.append(b[0])
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    c = move_left(tile, board)
    if c[0] != board:
        l.append(c[0])
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    d = move_right(tile, board)
    if d[0] != board:
        l.append(d[0])
    return l

# Successors for disjoint patterns

def successor(board1, tile):
    list = []
    # print board1
    o1 = twod_oned(board1)
    o = hash_tiles(o1)
    list1 = [1,2,3,4,5,6,7,8]
    list2 = [15,14,13,12,11,10,9]
    board = copy.deepcopy(board1)
    curr_board1 = tiles_dict[o][3]
    curr_board = copy.deepcopy(curr_board1)
    tile1 = tiles_dict[o][0]
    level = tiles_dict[o][1]
    a, path_element1 = move_up(tile, board)
    if a == curr_board:
        if (tile in list2 and path_element1 < tile) or (tile in list1 and path_element1 > tile):
            x = expand_next_element(path_element1, board)
            for i in range(len(x)):
                a1 = twod_oned(x[i])
                p = hash_tiles(a1)
                if p not in tiles_dict:
                    tiles_dict[p] = [path_element1, level+1, 0, x[i], board, 0]
                    list.append(x[i])
    else:
        a1 = twod_oned(a)
        p = hash_tiles(a1)
        if p not in tiles_dict:
            tiles_dict[p] = [tile, level+1, 0, a, board, 0]
            list.append(a)
        if position(tile, a) == goal_position(tile) and a not in list:
            list.append(a)
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    b, path_element2 = move_down(tile, board)
    if b == curr_board:
        if (tile in list2 and path_element2 < tile) or (tile in list1 and path_element2 > tile):
            y = expand_next_element(path_element2, board)
            for i in range(len(y)):
                b1 = twod_oned(y[i])
                q = hash_tiles(b1)
                if q not in tiles_dict:
                    tiles_dict[q] = [path_element2, level+1, 0, y[i], board, 0]
                    list.append(y[i])
    else:
        b1 = twod_oned(b)
        r = hash_tiles(b1)
        if r not in tiles_dict:
            tiles_dict[r] = [tile, level+1, 0, b, board, 0]
            list.append(b)
        if position(tile, b) == goal_position(tile) and b not in list:
            list.append(b)
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    c, path_element3 = move_left(tile, board)
    if c == curr_board:
        if (tile in list2 and path_element3 < tile) or (tile in list1 and path_element3 > tile):
            w = expand_next_element(path_element3, board)
            for i in range(len(w)):
                c1 = twod_oned(w[i])
                s = hash_tiles(c1)
                if s not in tiles_dict:
                    tiles_dict[s] = [path_element3, level+1, 0, w[i], board, 0]
                    list.append(w[i])
    else:
        c1 = twod_oned(c)
        s = hash_tiles(c1)
        if s not in  tiles_dict:
            tiles_dict[s] = [tile, level+1, 0, c, board, 0]
            list.append(c)
        if position(tile, c) == goal_position(tile) and c not in list:
            list.append(c)
    # board = [[5,0,0,0], [0,3,0,1], [0,0,0,9], [2,0,4,13]]
    d,path_element4 = move_right(tile, board)
    if d == curr_board:
        if (tile in list2 and path_element4 < tile) or (tile in list1 and path_element4 > tile):
            z = expand_next_element(path_element4, board)
            for i in range(len(z)):
                d1 = twod_oned(z[i])
                t = hash_tiles(d1)
                if t not in tiles_dict:
                    tiles_dict[t] = [path_element4, level+1, 0, z[i], board, 0]
                    list.append(z[i])
    else:
        d1 = twod_oned(d)
        t = hash_tiles(d1)
        if t not in tiles_dict:
            tiles_dict[t] = [tile, level+1, 0, d, board, 0]
            list.append(d)
        if position(tile, d) == goal_position(tile) and d not in list:
            list.append(d)
    return list

#position of a tile given a board
def position(tile, board):
    # board = [[5,10,14,7], [8,3,6,1], [15,0,12,9],[2,11,4,13]]
    for i in range(len(board)):
        for l in range(len(board[i])):
            if board[i][l] == tile:
                return i,l

#Goal position
def goal_position(tile):
    if (tile/4.0) == 1 or (tile/4.0) == 2 or (tile/4.0) == 3 or (tile/4.0) == 4:
        r1 = (tile/4 - 1)
        c1 = 3
    else:
        r1 = (tile/4)
        c1 = (tile % 4 - 1)
    return r1,c1

# Convert two Dimensional to one Dimensional array
def twod_oned(board):
    one_d = []
    w = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            one_d.append(board[i][j])
    return one_d

#hash function for each tile
def hash_tiles(state):
	  idx = 0
	  for i in range(len(state)):
	    val = state[i]
	    idx |= i << (val * 4)
	  return idx

#make patterns
def divide_board(board):
    par1 = [1,2,3,4,5,6,7,8]
    par2 = [15,14,13,12,11,10,9]
    board1 = [[0]*4 for _ in range(4)]
    board2 = [[0]*4 for _ in range(4)]
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] in par1:
                board1[i][j] = board[i][j]
            elif board[i][j] in par2:
                board2[i][j] = board[i][j]
    return [board1, board2]

def is_goal_board(board):
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] != 0:
                r1,r2 = position(board[i][j], board)
                c1,c2 = goal_position(board[i][j])
                if r1 != c1 or r2 != c2:
                    return 0
    return 1

def board_children(board1):
    board = board1
    b1 = twod_oned(board)
    b2 = hash_tiles(b1)
    l = []
    first_child = move_down_board(board)
    q1 = twod_oned(first_child)
    key1 = hash_tiles(q1)
    if key1 not in tiles_parent:
        tiles_parent[key1] = [first_child, board, tiles_parent[b2][2]+1]
        l.append(first_child)
    second_child = move_up_board(board)
    q2 = twod_oned(second_child)
    key2 = hash_tiles(q2)
    if key2 not in tiles_parent:
        tiles_parent[key2] = [second_child, board, tiles_parent[b2][2]+1]
        l.append(second_child)
    third_child = move_left_board(board)
    q3 = twod_oned(third_child)
    key3 = hash_tiles(q3)
    if key3 not in tiles_parent:
        tiles_parent[key3] = [third_child, board, tiles_parent[b2][2]+1]
        l.append(third_child)
    fourth_child = move_right_board(board)
    q4 = twod_oned(fourth_child)
    key4 = hash_tiles(q4)
    if key4 not in tiles_parent:
        tiles_parent[key4] = [fourth_child, board, tiles_parent[b2][2]+1]
        l.append(fourth_child)
    # l.extend([first_child, second_child, third_child, fourth_child])
    return l


#Find minimum moves for each pattern
def solve_tiles(b1, my_lists):
    global tiles_dict
    tiles_dict = {}
    local_list = []
    local_list.append(b1)
    states = []
    tiles_obj1 = boards(b1, my_lists, 0, 1)
    tiles_obj = copy.deepcopy(tiles_obj1)
    q = twod_oned(tiles_obj.id)
    key = hash_tiles(q)
    tiles_dict[key] = [tiles_obj.element, tiles_obj.depth, tiles_obj.visited, tiles_obj.id, 'root', 0]
    fringe = []
    fringe.append(tiles_obj.id)
    deep_level = 500000
    if my_lists == 1:
        my_list = [1,2,3,4,5,6,7,8]
    else:
        my_list = [15,14,13,12,11,10,9]

    while len(my_list) > 0:
        num = my_list.pop(0)
        e_exists = 0
        r = []
        for v in range(len(states)):
            v1,v2 = position(num, states[v][0])
            w1,w2 = goal_position(num)
            if deep_level == states[v][1]:
                if v1 == w1 and v2 == w2:
                    e_exists = 1
                    r.append(states[v])
        if e_exists == 1:
            temp_states = copy.deepcopy(states)
            for h in range(len(states)):
                if deep_level == states[h][1]:
                    u1,u2 = position(num, states[h][0])
                    e1,e2 = goal_position(num)
                    if u1 != e1 or u2 != e2:
                        temp_states.remove(states[h])
            states = temp_states
            continue

        if deep_level != 500000:
            fringe = []
            for v in range(len(states)):
                if states[v][1] == deep_level:
                    fringe.append(states[v][0])

        deep_level = 500000
        while len(fringe) > 0:
            g = fringe.pop(0)
            k = copy.deepcopy(g)
            p3,p4 = position(num, k)
            p5,p6 = goal_position(num)
            if p3 == p5 and p4==p6:
                l5 = twod_oned(k)
                l8 = hash_tiles(l5)
                deep_level = tiles_dict[l8][1]
                if [k, deep_level] not in states:
                    states.append([k,deep_level])
                if len(fringe) == 0:
                    break
            else:
                s = successor(k, num)
                s = [x for x in s if x not in local_list]
                local_list.extend(s)
                k1 = twod_oned(k)
                dict_key = hash_tiles(k1)
                tiles_dict[dict_key][2] = 1
                if tiles_dict[dict_key][1] > deep_level-1:
                    break
                for i in s:
                    p1,p2 = position(num, i)
                    p3,p4 = goal_position(num)
                    if p1 == p3 and p2 == p4:
                        l1 = twod_oned(i)
                        l4 = hash_tiles(l1)
                        deep_level = tiles_dict[l4][1]
                        if [i, tiles_dict[l4][1]] not in states:
                            states.append([i,tiles_dict[l4][1]])

                    else:
                        x1 = twod_oned(i)
                        x = hash_tiles(x1)
                        if tiles_dict[x][2] == 0:
                            fringe.append(i)
    return states


#Solve the problem based on heuristics
def solve_dfs():
    global check_consistency
    i_board = copy.deepcopy(i_board1)
    one = twod_oned(i_board)
    key = hash_tiles(one)
    tiles_parent[key] = [i_board,'root', 0]
    # local_list.append(i_board)
    # b = divide_board(i_board)
    if is_goal_board(i_board) == 1:
        return i_board
    s = board_children(i_board)
    fringe = []
    main_list = []
    main_list.append(i_board)
    # s = [x for x in s if x not in local_list]
    # local_list.extend(s)
    fringe.extend(s)
    # main_list.extend(s)
    # for i in range(len(s)):
    #     if s[i] not in main_list:
    #         main_list.append(s[i])
    while len(fringe) > 0:
        cost = 0
        min = 20000
        for i in range(len(fringe)):
            d1 = divide_board(fringe[i])
            s1 = solve_tiles(d1[0], 1)
            s2 = solve_tiles(d1[1], 9)
            k1 = twod_oned(fringe[i])
            k2 = hash_tiles(k1)
            # print s2,d1[1]
            t = s1[len(s1)-1][1] + s2[len(s2)-1][1]
            if check_consistency == 0:
                if t < min:
                    id1 = fringe[i]
                    min = t
            elif check_consistency == 1:
                if t < min:
                    id1 = fringe[i]
                    min = t
                    cost = tiles_parent[k2][1]
                elif t == min:
                    if tiles_parent[k2][1] < cost:
                        id1 = fringe[i]
                        cost = tiles_parent[k2][1]
        fringe.remove(id1)
        op = twod_oned(id1)
        op1 = hash_tiles(op)
        # print id1, min, tiles_parent[op1]
        # if is_goal_board(id1) == 1 or min == 0:
        #     print "Found it here", id1
        #     return 0
        c1 = board_children(id1)
        for x in range(len(c1)):
            if is_goal_board(c1[x]) == 1:
                # print "Found it there", c1[x],
                return c1[x]
            else:
                if c1[x] not in main_list:
                    main_list.append(c1[x])
                    fringe.append(c1[x])

# Check for inversions
def check_board1():
    counter = 0
    a= twod_oned(i_board1)
    while len(a) > 0:
        k = a.pop(0)
        for t in range(len(a)):
            if a[t] < k and a[t] != 0:
                counter += 1
    row, col = position(0, i_board1)
    return row+1+counter

def read_file():
    read_file = open(my_file, 'r')
    a = []
    for line in read_file:
        items = line.rstrip('\n').split(' ')
        u = [int(items[0]), int(items[1]), int(items[2]), int(items[3])]
        a.append(u)
    # b = [int[items[4]], int[items[5]], int[items[6]], int[items[7]]]
    # c = [int[items[8]], int[items[9]], int[items[10]], int[items[11]]]
    # d = [int[items[12]], int[items[13]], int[items[14]], int[items[15]]]
    return a

tiles_parent = {}
tiles_dict = {}
tiles_disjoint_parents = {}
count = 0
# print is_goal_board([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])
# [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 15, 12], [13, 0, 11, 14]]
# [[1,2,3,4],[9,10,11,12],[5,6,8,7],[13,15,14,0]]
my_file = sys.argv[1]
w = read_file()
intial_board = w
i_board1 = copy.deepcopy(intial_board)
 # There is consistency check so user can find for either optimal or sub optimal solution
#  . There is a trade-off between the optimal output and the time complexity for this heuristic which can be worked
# upon in the future.
check_consistency = 0
start_time = time.time()
# print "Position",position(0, i_board1)
check = check_board1()
if check % 2 != 0:
    print "Not solvable"
else:
    solve_dfs()
    print("--- %s seconds ---" % (time.time() - start_time))
    final_conv = twod_oned([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]])
    final_key = hash_tiles(final_conv)
    parent = tiles_parent[final_key][0]
    move = []
    while parent != 'root':
        # print parent
        final_conv = twod_oned(parent)
        final_key = hash_tiles(final_conv)
        a = move_up_board(parent)
        b = move_down_board(parent)
        c = move_left_board(parent)
        d = move_right_board(parent)
        # print a,b,c,d,parent
        parent = tiles_parent[final_key][1]
        if a == parent:
            move.append('U')
        elif b == parent:
            move.append('D')
        elif c == parent:
            move.append('L')
        elif d == parent:
            move.append('R')
        # print tiles_parent[final_key]
    ments = i_board1
    for n in range(len(move)):
        if move[len(move)-1-n] == 'D':
            ments = move_up_board(ments)
            # print ments, 'D'
        elif move[len(move)-1-n] == 'U':
            ments = move_down_board(ments)
            # print ments, 'U'
        elif move[len(move)-1-n] == 'L':
            ments = move_right_board(ments)
            # print ments, 'L'
        elif move[len(move)-1-n] == 'R':
            ments = move_left_board(ments)
            # print ments, 'R'
    # print move
    move1 = []
    for i in reversed(move):
        move1.append(i)
    print move1
count_steps = 0
def Search(board, EMPTY, BLACK, WHITE, isblack):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return 
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    #return RandomSearch(board, EMPTY)
    global count_steps
    count_steps += 1
    return AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack,count_steps)

def RandomSearch(board, EMPTY):
    # AI 的占位行为，随机选择一个位置落子
    # 在实现 alpha-beta 剪枝中不需要使用
    from random import randint
    ROWS = len(board)
    x = randint(0, ROWS - 1)
    y = randint(0, ROWS - 1)
    while board[x][y] != EMPTY:
        x = randint(0, ROWS - 1)
        y = randint(0, ROWS - 1)
    return x, y, 0

def AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack,count_steps):
    '''
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    EMPTY       空格在 board 中的表示，默认为 -1
    BLACK       黑棋在 board 中的表示，默认为 1
    WHITE       白棋在 board 中的表示，默认为 0
    isblack     bool 变量，表示当前是否轮到黑子落子
    ---------------返回---------------
    x           落子的 x 坐标（行数/第一维）
    y           落子的 y 坐标（列数/第二维）
    alpha       本层的 alpha 值
    '''
    # 请修改此函数，实现 alpha-beta 剪枝
    # =============你的代码=============
    ...
    #if count_steps == 1:
        #return 7,7,0
    coordinate = ()
    if isblack:
        coordinate,alpha = MinMax(1,1,3,float('-inf'),float('inf'),board,(0,0))
    return coordinate[0], coordinate[1], alpha

# 你可能还需要定义评价函数或者别的什么
# =============你的代码=============
...



# 以下为编写搜索和评价函数时可能会用到的函数，请看情况使用、修改和优化
# =============辅助函数=============
def MinMax(n,player,depth_limit,alpha,beta,board,coordinate):
    x = -1
    y = -1
    if n != 1:
        grade = analysis(board)
        if grade >= 9999:
            return (coordinate,grade)
    if n == depth_limit:
        grade = 0
        grade += analysis(board)
        return (coordinate,grade)
    if player == 1:
        next_state = get_successors(board,player,_coordinate_priority,-1)
        coordinate=()
        for x,y,state in next_state:
            if x == 5 and y == 9:
                temp = 0
            temp_alpha = MinMax(n+1,0,depth_limit,alpha,beta,state,(x,y))[1]
            if( temp_alpha > alpha):
                alpha = temp_alpha
                coordinate = (x,y)
            if beta <= alpha:
                break
        return (coordinate,alpha)
    else:
        next_state = get_successors(board,player,_coordinate_priority,-1)
        coordinate = ()
        for x,y,state in next_state:
            temp_beta = MinMax(n+1,1,depth_limit,alpha,beta,state,(x,y))[1]
            if temp_beta < beta:
                beta = temp_beta
                coordinate = (x,y)
            if beta <= alpha:
                break
        return (coordinate,beta)
priority_board = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                  [0,1,2,2,2,2,2,2,2,2,2,2,2,1,0],
                  [0,1,2,3,3,3,3,3,3,3,3,3,2,1,0],
                  [0,1,2,3,4,4,4,4,4,4,4,3,2,1,0],
                  [0,1,2,3,4,5,5,5,5,5,4,3,2,1,0],
                  [0,1,2,3,4,5,6,6,6,5,4,3,2,1,0],
                  [0,1,2,3,4,5,6,7,6,5,4,3,2,1,0],
                  [0,1,2,3,4,5,6,6,6,5,4,3,2,1,0],
                  [0,1,2,3,4,5,5,5,5,5,4,3,2,1,0],
                  [0,1,2,3,4,4,4,4,4,4,4,3,2,1,0],
                  [0,1,2,3,3,3,3,3,3,3,3,3,2,1,0],
                  [0,1,2,2,2,2,2,2,2,2,2,2,2,1,0],
                  [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
                  [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
def get_grade(line):
    black_wulian_list = [1,1,1,1,1]
    black_wulian_num = 0
    white_wulian_list = [0,0,0,0,0]
    white_wulian_num = 0
    black_lianchongsi_list1 = [-1,1,1,1,1,0]
    black_lianchongsi_list2 = [0,1,1,1,1,-1]
    black_tiaochongsi_list1 = [1,1,1,-1,1]
    black_tiaochongsi_list2 = [1,1,-1,1,1]
    black_tiaochongsi_list3 = [1,-1,1,1,1]
    black_chongsi_num = 0
    white_lianchongsi_list1 = [-1,0,0,0,0,1]
    white_lianchongsi_list2 = [1,0,0,0,0,-1]
    white_tiaochongsi_list1 = [0,0,0,-1,0]
    white_tiaochongsi_list2 = [0,0,-1,0,0]
    white_tiaochongsi_list3 = [0,-1,0,0,0]
    white_chongsi_num = 0
    black_huosi_list = [-1,1,1,1,1,-1]
    white_huosi_list = [-1,0,0,0,0,-1]
    black_huosi_num = 0
    white_huosi_num = 0
    black_lianhuosan_list = [-1,1,1,1,-1]
    black_tiaohuosan_list1 = [-1,1,1,-1,1,-1]
    black_tiaohuosan_list2 = [-1,1,-1,1,1,-1]
    black_huosan_num = 0
    white_lianhuosan_list = [-1,0,0,0,-1]
    white_tiaohuosan_list1 = [-1,0,0,-1,0,-1]
    white_tiaohuosan_list2 = [-1,0,-1,0,0,-1]
    white_huosan_num = 0
    black_lianer_list = [-1,1,1,-1]
    black_tiaoer_list = [-1,1,-1,1,-1]
    black_datiaoer_list = [-1,1,-1,-1,1,-1]
    black_huoer_num = 0
    white_lianer_list = [-1,0,0,-1]
    white_tiaoer_list = [-1,0,0,0,-1]
    white_datiaoer_list = [-1,0,-1,-1,0,-1]
    white_huoer_num = 0
    black_chonger_list1 = [0,1,1,-1]
    black_chonger_list2 = [-1,0,0,1]
    black_chonger_num = 0
    white_chonger_list1 = [1,0,0,-1]
    white_chonger_list2 = [-1,0,0,1]
    white_chonger_num = 0
    black_miansan_list1 = [1,-1,1,-1,1]
    black_miansan_list2 = [0,1,1,1]
    black_miansan_list3 = [1,1,1,0]
    black_miansan_list4 = [0,1,1,-1,1]
    black_miansan_num = 0
    white_miansan_list1 = [0,-1,0,-1,0]
    white_miansan_list2 = [1,0,0,0]
    white_miansan_list3 = [0,0,0,1]
    white_miansan_list4 = [1,0,0,-1,0]
    white_miansan_num = 0
    if len(line) >= 3:
        if line[0:3] == [1,1,-1]:
            black_chonger_num += 1
        elif line[0:3] == [0,0,-1]:
            white_chonger_num += 1
        if line[len(line)-3:len(line)] == [-1,1,1]:
            black_chonger_num += 1
        elif line[len(line)-3:len(line)] == [-1,0,0]:
            white_chonger_num += 1
    if len(line) >= 5:
        if line[0:5] == [1,1,1,1,-1] :
            black_chongsi_num += 1
        if line[len(line)-5:len(line)] == [-1,1,1,1,1]:
            black_chongsi_num += 1
        if line[0:5] == [0,0,0,0,-1]:
            white_chongsi_num += 1
        if line[len(line)-5:len(line)] == [-1,0,0,0,0]:
            white_chongsi_num += 1
    for i in range(0,len(line)-5,1):
        if line[i:i+4] == black_lianer_list:
            black_huoer_num += 2
        elif line[i:i+4] == white_lianer_list:
            white_huoer_num += 2
        elif line[i:i+4] == black_chonger_list1 or line[i:i+4] == black_chonger_list2:
            black_chonger_num += 1
        elif line[i:i+4] == white_chonger_list1 or line[i:i+4] == white_chonger_list2:
            white_chonger_num += 1
        elif line[i:i+4] == black_miansan_list2 or line[i:i+4] == black_miansan_list3:
            black_miansan_num += 1
        elif line[i:i+4] == white_miansan_list2 or line[i:i+4] == white_miansan_list3:
            white_miansan_num += 1
        elif line[i:i+5] == black_wulian_list:
            black_wulian_num  += 1
        elif line[i:i+5] == white_wulian_list:
            white_wulian_num  += 1
        elif line[i:i+5] == black_tiaochongsi_list1 or line[i:i+5] == black_tiaochongsi_list2 or line[i:i+5] == black_tiaochongsi_list3:
            black_chongsi_num += 1
        elif line[i:i+5] == white_tiaochongsi_list1 or line[i:i+5] == white_tiaochongsi_list2 or line[i:i+5] == white_tiaochongsi_list3:
            white_chongsi_num += 1
        elif line[i:i+5] == black_lianhuosan_list:
            black_huosan_num += 1
            #print("lianhuosan")
        elif line[i:i+5] == white_lianhuosan_list:
            white_huosan_num += 1
        elif line[i:i+5] == black_tiaoer_list:
            black_huoer_num += 1
        elif line[i:i+5] == white_tiaoer_list:
            white_huoer_num += 1
        elif line[i:i+5] == black_miansan_list1 or line[i:i+5] == black_miansan_list4:
            black_miansan_num += 1
        elif  line[i:i+5] == white_miansan_list1 or line[i:i+5] == white_miansan_list4:
            white_miansan_num += 1
        if i < len(line)-6:
            if line[i:i+6] == black_lianchongsi_list1 or line[i:i+6] == black_lianchongsi_list2:
                black_chongsi_num += 1
            elif line[i:i+6] == white_lianchongsi_list1 or line[i:i+6] == white_lianchongsi_list2:
                white_chongsi_num += 1
            elif line[i:i+6] == black_huosi_list:
                black_huosi_num += 1
            elif line[i:i+6] == white_huosi_list:
                white_huosi_num += 1
            elif line[i:i+6] == black_tiaohuosan_list1 or line[i:i+6] == black_tiaohuosan_list2:
                black_huosan_num += 1
                #print(i,"tiaohuoshan")
            elif line[i:i+6] == white_tiaohuosan_list1 or line[i:i+6] == white_tiaohuosan_list2:
                white_huosan_num += 1
            elif line[i:i+6] == black_datiaoer_list:
                black_huoer_num += 1
            elif line[i:i+6] == white_datiaoer_list:
                white_huoer_num += 1
    if len(line)>=4:
        if line[len(line)-4:len(line)] == black_lianer_list:
            black_huoer_num += 2
        elif line[len(line)-4:len(line)] == white_lianer_list:
            white_huoer_num += 2
        elif line[len(line)-4:len(line)] == black_miansan_list2 or line[len(line)-4:len(line)] == black_miansan_list3:
            black_miansan_num += 1
        elif  line[len(line)-4:len(line)] == white_miansan_list2 or line[len(line)-4:len(line)] == white_miansan_list3:
            white_miansan_num += 1
    return (black_wulian_num,white_wulian_num,black_huosi_num,white_huosi_num,black_chongsi_num,white_chongsi_num,black_huosan_num,white_huosan_num,black_huoer_num,white_huoer_num,black_chonger_num,white_chonger_num,black_miansan_num,white_miansan_num)
def get_vertical_board(board):
    for i in range(15):
        line = []
        for j in range(15):
            line.append(board[j][i])
        yield(line)
def get_left_board(board):
    for i in range(15):
        line = []
        row = i
        column = 0
        while row >= 0:
            line.append(board[row][column])
            row -= 1
            column += 1
        yield(line)
    for i in range(1,15):
        line = []
        column = i
        row = 14
        while column <= 14:
            line.append(board[row][column])
            row -= 1
            column += 1
        yield(line)

def get_right_board(board):
    for i in range(15):
        line = []
        column = 14
        row = i
        while column >= 0:
            line.append(board[row][column])
            column -= 1
            row -= 1
        yield(line)
    for i in range(1,15):
        line = []
        row = i
        column = 0
        while row <= 14:
            line.append(board[row][column])
            row += 1
            column += 1
        yield(line)
def analysis(board):
    grade = 0
    black_wulian_num = 0
    white_wulian_num = 0
    black_huosi_num = 0
    white_huosi_num = 0
    black_chongsi_num = 0
    white_chongsi_num = 0
    black_huosan_num = 0
    white_huosan_num = 0
    black_huoer_num = 0
    white_huoer_num = 0
    black_chonger_num = 0
    white_chonger_num = 0
    black_miansan_num = 0
    white_miansan_num = 0
    for line in board:
        grade_tuple = get_grade(line)
        black_wulian_num += grade_tuple[0]
        white_wulian_num += grade_tuple[1]
        black_huosi_num += grade_tuple[2]
        white_huosi_num += grade_tuple[3]
        black_chongsi_num += grade_tuple[4]
        white_chongsi_num += grade_tuple[5]
        black_huosan_num += grade_tuple[6]
        white_huosan_num += grade_tuple[7]
        black_huoer_num += grade_tuple[8]
        white_huoer_num += grade_tuple[9]
        black_chonger_num += grade_tuple[10]
        white_chonger_num += grade_tuple[11]
        black_miansan_num += grade_tuple[12]
        white_miansan_num += grade_tuple[13]
    vertical_board = get_vertical_board(board)
    for line in vertical_board:
        grade_tuple = get_grade(line)
        black_wulian_num += grade_tuple[0]
        white_wulian_num += grade_tuple[1]
        black_huosi_num += grade_tuple[2]
        white_huosi_num += grade_tuple[3]
        black_chongsi_num += grade_tuple[4]
        white_chongsi_num += grade_tuple[5]
        black_huosan_num += grade_tuple[6]
        white_huosan_num += grade_tuple[7]
        black_huoer_num += grade_tuple[8]
        white_huoer_num += grade_tuple[9]
        black_chonger_num += grade_tuple[10]
        white_chonger_num += grade_tuple[11]
        black_miansan_num += grade_tuple[12]
        white_miansan_num += grade_tuple[13]
    left_board = get_left_board(board)
    for line in left_board:
        grade_tuple = get_grade(line)
        black_wulian_num += grade_tuple[0]
        white_wulian_num += grade_tuple[1]
        black_huosi_num += grade_tuple[2]
        white_huosi_num += grade_tuple[3]
        black_chongsi_num += grade_tuple[4]
        white_chongsi_num += grade_tuple[5]
        black_huosan_num += grade_tuple[6]
        white_huosan_num += grade_tuple[7]
        black_huoer_num += grade_tuple[8]
        white_huoer_num += grade_tuple[9]
        black_chonger_num += grade_tuple[10]
        white_chonger_num += grade_tuple[11]
        black_miansan_num += grade_tuple[12]
        white_miansan_num += grade_tuple[13]
    right_board = get_right_board(board)
    for line in right_board:
        grade_tuple = get_grade(line)
        black_wulian_num += grade_tuple[0]
        white_wulian_num += grade_tuple[1]
        black_huosi_num += grade_tuple[2]
        white_huosi_num += grade_tuple[3]
        black_chongsi_num += grade_tuple[4]
        white_chongsi_num += grade_tuple[5]
        black_huosan_num += grade_tuple[6]
        white_huosan_num += grade_tuple[7]
        black_huoer_num += grade_tuple[8]
        white_huoer_num += grade_tuple[9]
        black_chonger_num += grade_tuple[10]
        white_chonger_num += grade_tuple[11]
        black_miansan_num += grade_tuple[12]
        white_miansan_num += grade_tuple[13]        
    """if black_chongsi_num >= 2:
        black_huosi_num += 1
    if white_chongsi_num >= 2:
        white_huosi_num += 1
    if black_wulian_num > 0:
        return 9999
    elif white_wulian_num > 0:
        return -9999
    elif black_huosi_num > 0:
        return 9990
    #elif black_chongsi_num > 0:
        return 9980             #最后一层为白棋落子，才有此判断
    elif black_chongsi_num > 0 and white_huosan_num > 0:
        return 9985
    elif white_huosi_num > 0: #最后一层的落子方影响优先级
        return -9970"""
    if black_huosan_num > 1 and white_chongsi_num == 0 and white_huosan_num == 0 and white_miansan_num == 0:
        grade +=  15000
    """elif white_chongsi_num > 0 and white_huosan_num > 0:
        return -9960"""
    if white_huosan_num > 1 and black_chongsi_num == 0 and black_huosan_num == 0:
        grade +=  -15000
    """if black_huosan_num > 1:
        grade += 2000
    elif black_huosan_num == 1:
        grade += 200
    if white_huosan_num > 1:
        grade += 500
    elif white_huosan_num == 1:
        grade += 100"""
    grade += black_miansan_num*200 - white_miansan_num*200+ black_huoer_num*50-white_huoer_num*50+black_chonger_num*20-white_chonger_num*20+black_huosan_num*2000-white_huosan_num*1000+black_chongsi_num*50000-white_chongsi_num*100000+black_huosi_num*30000-white_huosi_num*6000+black_wulian_num*9999999- white_wulian_num*9999999
    for i in range(15):
        for j in range(15):
            if board[i][j] == 1:
                grade += priority_board[i][j]
            elif board[i][j] == 0:
                grade += -priority_board[i][j]
    #if grade >= 9999:
        #print(board)
    return grade
def _coordinate_priority(coordinate):
    x= coordinate[0]
    y = coordinate[1]
    return -priority_board[x][y]
def near_center(board,x,y):
    start_X = x - 2
    end_x = x + 2
    start_y = y -2
    end_y = y + 2
    if start_X < 0:
        start_X = 0
    if end_x > len(board)-1:
        end_x = len(board)-1
    if start_y  < 0:
        start_y = 0
    if end_y > len(board)-1:
        end_y = len(board)-1
    for i in range(start_X,end_x+1):
        for j in range(start_y,end_y+1):
            if board[i][j] != -1:
                return True
    return False
def get_successors(board, color, priority=_coordinate_priority, EMPTY=-1):
    '''
    返回当前状态的所有后继（默认按坐标顺序从左往右，从上往下）
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    color       当前轮到的颜色
    EMPTY       空格在 board 中的表示，默认为 -1
    priority    判断落子坐标优先级的函数（结果为小的优先）
    ---------------返回---------------
    一个生成器，每次迭代返回一个的后继状态 (x, y, next_board)
        x           落子的 x 坐标（行数/第一维）
        y           落子的 y 坐标（列数/第二维）
        next_board  后继棋盘
    '''
    # 注意：生成器返回的所有 next_board 是同一个 list！
    from copy import deepcopy
    next_board = deepcopy(board)
    ROWS = len(board)
    idx_list = [(x, y) for x in range(15) for y in range(15)]
    idx_list.sort(key=priority)
    #print(idx_list)
    for x, y in idx_list:
        if board[x][y] == EMPTY and near_center(board,x,y):
            next_board[x][y] = color
            #print(x,y)
            yield (x, y, next_board)
            next_board[x][y] = EMPTY

# 这是使用 successors 函数的一个例子，打印所有后继棋盘
def _test_print_successors():
    '''
    棋盘：
      0 y 1   2
    0 1---+---1
    x |   |   |
    1 +---0---0
      |   |   |
    2 +---+---1
    本步轮到 1 下
    '''
    board = [
        [ 1, -1,  1],
        [-1,  0,  0],
        [-1, -1,  1]]
    EMPTY = -1
    next_states = get_successors(board, 1)
    for x, y, state in next_states:
        print(x, y, state)
    # 输出：
    # 0 1 [[1, 1, 1], [-1, 0, 0], [-1, -1, 1]]
    # 1 0 [[1, -1, 1], [1, 0, 0], [-1, -1, 1]]
    # 2 0 [[1, -1, 1], [-1, 0, 0], [1, -1, 1]]
    # 2 1 [[1, -1, 1], [-1, 0, 0], [-1, 1, 1]]

def get_next_move_locations(board, EMPTY=-1):
    '''
    获取下一步的所有可能落子位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    EMPTY       空格在 board 中的表示，默认为 -1
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表一个可下的坐标
    '''
    next_move_locations = []
    ROWS = len(board)
    for x in range(ROWS):
        for y in range(ROWS):
            if board[x][y] != EMPTY:
                next_move_locations.append((x,y))
    return next_move_locations

def get_pattern_locations(board, pattern):
    '''
    获取给定的棋子排列所在的位置
    ---------------参数---------------
    board       当前的局面，是 15×15 的二维 list，表示棋盘
    pattern     代表需要找的排列的 tuple
    ---------------返回---------------
    一个由 tuple 组成的 list，每个 tuple 代表在棋盘中找到的一个棋子排列
        tuple 的第 0 维     棋子排列的初始 x 坐标（行数/第一维）
        tuple 的第 1 维     棋子排列的初始 y 坐标（列数/第二维）
        tuple 的第 2 维     棋子排列的方向，0 为向下，1 为向右，2 为右下，3 为左下；
                            仅对不对称排列：4 为向上，5 为向左，6 为左上，7 为右上；
                            仅对长度为 1 的排列：方向默认为 0
    ---------------示例---------------
    对于以下的 board（W 为白子，B为黑子）
      0 y 1   2   3   4   ...
    0 +---W---+---+---+-- ...
    x |   |   |   |   |   ...
    1 +---+---B---+---+-- ...
      |   |   |   |   |   ...
    2 +---+---+---W---+-- ...
      |   |   |   |   |   ...
    3 +---+---+---+---+-- ...
      |   |   |   |   |   ...
    ...
    和要找的 pattern (WHITE, BLACK, WHITE)：
    函数输出的 list 会包含 (0, 1, 2) 这一元组，代表在 (0, 1) 的向右下方向找到了
    一个对应 pattern 的棋子排列。
    '''
    ROWS = len(board)
    DIRE = [(1, 0), (0, 1), (1, 1), (1, -1)]
    pattern_list = []
    palindrome = True if tuple(reversed(pattern)) == pattern else False
    for x in range(ROWS):
        for y in range(ROWS):
            if pattern[0] == board[x][y]:
                if len(pattern) == 1:
                    pattern_list.append((x, y, 0))
                else:
                    for dire_flag, dire in enumerate(DIRE):
                        if _check_pattern(board, ROWS, x, y, pattern, dire[0], dire[1]):
                            pattern_list.append((x, y, dire_flag))
                    if not palindrome:
                        for dire_flag, dire in enumerate(DIRE):
                            if _check_pattern(board, ROWS, x, y, pattern, -dire[0], -dire[1]):
                                pattern_list.append((x, y, dire_flag + 4))
    return pattern_list

# get_pattern_locations 调用的函数
def _check_pattern(board, ROWS, x, y, pattern, dx, dy):
    for goal in pattern[1:]:
        x, y = x + dx, y + dy
        if x < 0 or y < 0 or x >= ROWS or y >= ROWS or board[x][y] != goal:
            return False
    return True

def count_pattern(board, pattern):
    # 获取给定的棋子排列的个数
    return len(get_pattern_locations(board, pattern))

def is_win(board, color, EMPTY=-1):
    # 检查在当前 board 中 color 是否胜利
    pattern1 = (color, color, color, color, color)          # 检查五子相连
    pattern2 = (EMPTY, color, color, color, color, EMPTY)   # 检查「活四」
    return count_pattern(board, pattern1) + count_pattern(board, pattern2) > 0

# 这是使用以上函数的一个例子
def _test_find_pattern():
    '''
    棋盘：
      0 y 1   2   3   4   5
    0 1---+---1---+---+---+
    x |   |   |   |   |   |
    1 +---0---0---0---0---+ ... 此行有 0 的「活四」
      |   |   |   |   |   |
    2 +---+---1---+---+---1
      |   |   |   |   |   |
    3 +---+---+---+---0---+
      |   |   |   |   |   |
    4 +---+---+---1---0---1
      |   |   |   |   |   |
    5 +---+---+---+---+---+
    '''
    board = [
        [ 1, -1,  1, -1, -1, -1],
        [-1,  0,  0,  0,  0, -1],
        [-1, -1,  1, -1, -1,  1],
        [-1, -1, -1, -1,  0, -1],
        [-1, -1, -1,  1,  0,  1],
        [-1, -1, -1, -1, -1, -1]]
    pattern = (1, 0, 1)
    pattern_list = get_pattern_locations(board, pattern)
    assert pattern_list == [(0, 0, 2), (0, 2, 0), (2, 5, 3), (4, 3, 1)]
        # (0, 0) 处有向右下的 pattern
        # (0, 2) 处有向下方的 pattern
        # (2, 5) 处有向左下的 pattern
        # (4, 3) 处有向右方的 pattern
    assert count_pattern(board, (1,)) == 6
        # 6 个 1
    assert count_pattern(board, (1, 0)) == 13
        # [(0, 0, 2), (0, 2, 0), (0, 2, 2), (0, 2, 3), (2, 2, 4), 
        #  (2, 2, 6), (2, 2, 7), (2, 5, 3), (2, 5, 6), (4, 3, 1), 
        #  (4, 3, 7), (4, 5, 5), (4, 5, 6)]
    assert is_win(board, 1) == False
        # 1 没有达到胜利条件
    assert is_win(board, 0) == True
        # 0 有「活四」，胜利

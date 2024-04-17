import time
# coding:utf-8
from collections import defaultdict

def Search(board, EMPTY, BLACK, WHITE, isblack):
    # 目前 AI 的行为是随机落子，请实现 AlphaBetaSearch 函数后注释掉现在的 return 
    # 语句，让函数调用你实现的 alpha-beta 剪枝
    #return RandomSearch(board, EMPTY)
    return AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack)

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
search_nodes = 0
def AlphaBetaSearch(board, EMPTY, BLACK, WHITE, isblack):
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
    global search_nodes
    search_nodes = 0
    coordinate = ()
    depth_limit = 5
    all_space = get_space_num(board)
    all_nodes = 1
    for i in range(depth_limit):
        all_space -= i
        all_nodes *= all_space
    start_time = time.time()
    grade = analysis_board(board)
    if isblack:
        coordinate,alpha = MinMax(0,1,depth_limit,float('-inf'),float('inf'),board,board,(0,0),grade)
    end_time = time.time()
    print("used time: %f"%(end_time-start_time))
    print("search_nodes:%d   all_nodes:%d"%(search_nodes,all_nodes))
    print("剪枝比:%.10f"%((all_nodes-search_nodes)/all_nodes))
    return coordinate[0], coordinate[1], alpha

# 你可能还需要定义评价函数或者别的什么
# =============你的代码=============

# ====ac自动机算法====
class Node(object):
    """
    node
    """
    def __init__(self, str='', is_root=False):
        self._next_p = {}
        self.fail = None
        self.is_root = is_root
        self.str = str
        self.parent = None

    def __iter__(self):
        return iter(self._next_p.keys())

    def __getitem__(self, item):
        return self._next_p[item]

    def __setitem__(self, key, value):
        _u = self._next_p.setdefault(key, value)
        _u.parent = self

    def __repr__(self):
        return "<Node object '%s' at %s>" % \
               (self.str, object.__repr__(self)[1:-1].split('at')[-1])

    def __str__(self):
        return self.__repr__()


class AhoCorasick(object):
    """
    Ac object
    """
    def __init__(self, *words):
        self.words_set = set(words)
        self.words = list(self.words_set)
        self.words.sort(key=lambda x: len(x))
        self._root = Node(is_root=True)
        self._node_meta = defaultdict(set)
        self._node_all = [(0, self._root)]
        _a = {}
        for word in self.words:
            for w in word:
                _a.setdefault(w, set())
                _a[w].add(word)

        def node_append(keyword):
            assert len(keyword) > 0
            _ = self._root
            for _i, k in enumerate(keyword):
                node = Node(k)
                if k in _:
                    pass
                else:
                    _[k] = node
                    self._node_all.append((_i+1, _[k]))
                if _i >= 1:
                    for _j in _a[k]:
                        if keyword[:_i+1].endswith(_j):
                            self._node_meta[id(_[k])].add((_j, len(_j)))
                _ = _[k]
            else:
                if _ != self._root:
                    self._node_meta[id(_)].add((keyword, len(keyword)))

        for word in self.words:
            node_append(word)
        self._node_all.sort(key=lambda x: x[0])
        self._make()

    def _make(self):
        """
        build ac tree
        :return:
        """
        for _level, node in self._node_all:
            if node == self._root or _level <= 1:
                node.fail = self._root
            else:
                _node = node.parent.fail
                while True:
                    if node.str in _node:
                        node.fail = _node[node.str]
                        break
                    else:
                        if _node == self._root:
                            node.fail = self._root
                            break
                        else:
                            _node = _node.fail

    def search(self, content, with_index=False):
        result = set()
        node = self._root
        index = 0
        for i in content:
            while 1:
                if i not in node:
                    if node == self._root:
                        break
                    else:
                        node = node.fail
                else:
                    for keyword, keyword_len in self._node_meta.get(id(node[i]), set()):
                        if not with_index:
                            result.add(keyword)
                        else:
                            result.add((keyword, (index - keyword_len + 1, index + 1)))
                    node = node[i]
                    break
            index += 1
        return result
#====ac自动机算法结束====

# 最小最大搜索算法，同时使用了alpha-beta剪枝
# 参数意义：
# n:当前递归层数，起始为0
# player:本层下棋方，初始为1，即黑棋先手
# depth_limit:层数的最大限制，可以更改
# alpha,beta:用于剪枝
# old_board:上一层递归的棋盘（上层未落子）  board:本层的棋盘（上层已落子）
# grade:总分数
def MinMax(n, player, depth_limit, alpha, beta, old_board, board, coordinate, grade):
    global search_nodes
    # 如果当前层数已经达到限制，计算分数并返回
    if n == depth_limit:
        # 减去落子坐标不落子的分数
        grade -= analysis_point((coordinate[0], coordinate[1], old_board))
        # 加上落子坐标落子的分数，其实这两行代码就是为了得到落子坐标处的净增加分数
        grade += analysis_point((coordinate[0], coordinate[1], board))
        return (coordinate, grade)

    if n != 0:
        # 如果不在第0层，则需要计算净分数
        grade -= analysis_point((coordinate[0], coordinate[1], old_board))
        grade += analysis_point((coordinate[0], coordinate[1], board))
        if abs(grade) > 35000:
            # 总分数绝对值大于35000，则该落子产生五连，继续搜索已经没有意义，所以返回分数
            return (coordinate, grade)
    #print("第%d层"%n)
    if player == 1:
        # 黑棋落子
        next_state = get_successors(board, player, -1)
        coordinate = ()
        for x, y, state in next_state:
            search_nodes += 1
            # 该落子的返回分数
            temp_alpha = MinMax(n + 1, 0, depth_limit, alpha, beta, board, state, (x, y), grade)[1]
            #if n == 0:
                #print("x:%d,y:%d,garde:%d"%(x,y,temp_alpha),end="   ")
            if temp_alpha > alpha:
                # 若大于alpha，则更新alpha，并记录坐标
                alpha = temp_alpha
                coordinate = (x, y)
            if beta <= alpha:
                # beta小于等于alpha则发生剪枝
                break
        return (coordinate, alpha)
    else:
        # 白棋落子，以下代码同黑棋落子
        next_state = get_successors(board, player, -1)
        coordinate = ()
        for x, y, state in next_state:
            search_nodes += 1
            temp_beta = MinMax(n + 1, 1, depth_limit, alpha, beta, board, state, (x, y), grade)[1]
            #print(x,y,temp_beta,end=" ")
            if temp_beta < beta:
                beta = temp_beta
                coordinate = (x, y)
            if beta <= alpha:
                break
        return (coordinate, beta)

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
grade_dict = {"#####":50000,' #### ':4320, # #代表黑棋，*代表白棋，‘ ’空格代表无棋
            ' ###  ':720,'  ### ':720,
            ' ## # ':720,' # ## ':720,
            '#### ':720,' ####':720,
            '## ##':720,'# ###':720,
            '### #':720,'  ##  ':120,
            '  # # ':120,' # #  ':120,
            '   #  ':20,'  #   ':20,
            "*****":-50000,' **** ':-4320,
            ' ***  ':-720, '  *** ':-720,
            ' ** * ':-720,' * ** ':-720,
            '**** ':-720,' ****':-720,
            '** **':-720,'* ***':-720,
            '*** *':-720,'  **  ':-120,
            '  * * ':-120, ' * *  ':-120,
            '   *  ':-20, '  *   ':-20}
ac = AhoCorasick("#####",' #### ', ' ###  ', '  ### ',' ## # ',' # ## ', '#### ',' ####', '## ##','# ###', '### #','  ##  ', '  # # ', ' # #  ','   #  ', '  #   ',"*****",' **** ', ' ***  ', '  *** ',' ** * ',' * ** ', '**** ',' ****', '** **','* ***', '*** *','  **  ', '  * * ', ' * *  ','   *  ', '  *   ')
def get_space_num(board):
    count = 0
    for i in range(15):
        for j in range(15):
            if board[i][j] == -1:
                count += 1
    return count
def get_grade(line,ac):
    # 主串
    text = "" 
    # 遍历传入评分函数的列表
    for i in line:
        # 空
        if i == -1:
            text += ' '
        # 黑棋
        elif i == 1:
            text += '#'
        # 白棋
        elif i == 0:
            text += '*'
    # 利用ac算法，得到该主串中所包含的模式串，即上述评分规则的棋形
    grade_set = ac.search(text)
    grade = 0
    # 在评分规则的字典中查出对应分数
    for i in grade_set:
        grade += grade_dict[i]
    # 返回分数
    return grade
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
        while row >= 0:
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
def vaild(x,y):
    return x>=0 and x <=14 and y>=0 and y <=14
def analysis_board(board):
    grade = 0

    for line in board:
        grade += get_grade(line,ac)
    vertical_board = get_vertical_board(board)
    for line in vertical_board:
        grade += get_grade(line,ac)
    left_board = get_left_board(board)
    for line in left_board:
        grade += get_grade(line,ac)
    right_board = get_right_board(board)
    for line in right_board:
        grade += get_grade(line,ac)
    return grade
def analysis_point(coordinate):
    # 坐标x
    x = coordinate[0]
    # 坐标y
    y = coordinate[1]
    # 棋盘
    board = coordinate[2]
    grade = 0
    # 得到横行的分数
    grade += get_grade(board[x],ac)
    line = []
    # 得到竖列
    for i in range(15):
        line.append(board[i][y])
    # 得到竖列的分数
    grade += get_grade(line,ac)
    line.clear()
    temp_x = x + y
    temp_y = 0
    # 得到右斜列
    while temp_x >= 0:
        if vaild(temp_x,temp_y):
            line.append(board[temp_x][temp_y])
        temp_x -= 1
        temp_y += 1
    # 得到右斜列分数
    grade += get_grade(line,ac)
    line.clear()
    temp_x = x -y
    temp_y = 0
    # 得到左斜列
    while temp_x <= 14:
        if vaild(temp_x,temp_y):
            line.append(board[temp_x][temp_y])
        temp_x += 1
        temp_y += 1
    # 得到左斜列分数
    grade += get_grade(line,ac)
    # 返回分数
    return grade
def _coordinate_priority(coordinate):
    x= coordinate[0]
    y = coordinate[1]
    return -priority_board[x][y]
def _heuristic_priority(state:tuple):
    # 坐标x
    x = state[0]
    # 坐标y
    y = state[1]
    # 棋局
    board = state[2]
    # 当前落子颜色
    color = state[3]
    # 如果当前坐标附近两格没有棋子，则该棋子不可能是目标落子点
    if not near_center(board,x,y):
        return -priority_board[x][y] #返回棋子的优先级的相反数，越在棋盘中间越高，主要是为了完成ai的第一步落子
    grade = 0
    # 减去未修改棋局时的分数
    grade -= analysis_point((x,y,board))
    flag = True
    # 如果当前坐标为空，则落子
    if board[x][y] != -1: 
        flag = False
    if flag:
        board[x][y] = color
    # 加上落子之后的分数，则grade为落子增加的净分数
    grade += analysis_point((x,y,board))
    if flag:
        board[x][y] = -1
    # 如果为黑棋，则返回分数的相反数，因为黑棋为Max节点，优先返回分数高的节点，而排序是从小到大的，所以此处取反
    if color == 1:
        return -grade
    # 如果为白棋，则直接返回分数
    else:
        return grade
def near_center(board,x,y):
    # 检测该坐标两格以内
    start_X = x - 2
    end_x = x + 2
    start_y = y -2
    end_y = y + 2
    # 排除一些边界情况
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
            # 如果不为空，返回True，说明附近有棋子
            if board[i][j] != -1:
                return True
    # 否则返回False，说明附近没有棋子
    return False

# 以下为编写搜索和评价函数时可能会用到的函数，请看情况使用、修改和优化
# =============辅助函数=============

def get_successors(board, color, EMPTY=-1):
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
    idx_list = [(x, y,board,color) for x in range(15) for y in range(15)]
    idx_list.sort(key=_heuristic_priority)
    #print(idx_list)
    count = 0
    for x, y,state,who in idx_list:
        if board[x][y] == EMPTY:
            next_board[x][y] = color
            count += 1
            #print(x,y)
            yield (x, y, next_board)
            next_board[x][y] = EMPTY
        if count >10:
            break
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

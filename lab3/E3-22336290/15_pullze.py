import queue
import time
pullze = []
goal = []
directions = [(1,0),(-1,0),(0,1),(0,-1)]
extened_nodes = 0
closed = dict()
parent = {}
new_limit = 0
flag = False
goal_state = ()
num = 1
for i in range(4):
    row = input().split()
    goal_row = []
    for j in range(4):
        row[j] = int(row[j])
        goal_row.append(num)
        num += 1
    if i == 3:
        goal_row[-1] = 0
    pullze.append(tuple(row))
    goal.append(tuple(goal_row))
pullze = tuple(pullze)
goal = tuple(goal)
steps = 0
actions = []
steps_pullze = []
def is_goal(state):
    return state == goal
def is_vaild(move_block):
    if move_block[0] < 4 and move_block[0] >= 0 and move_block[1] <4 and move_block[1] >= 0:
        return True
    return False
def change_to_tuple(pullze:list) ->tuple:
    for i in range(4):
        pullze[i] = tuple(pullze[i])
    pullze = tuple(pullze)
    return pullze
def change_to_list(pullze:tuple) ->list:
    pullze = list(pullze)
    for i in range(4):
        pullze[i] = list(pullze[i])
    pullze = list(pullze)
    return pullze
def find_zero_pos(pullze:tuple) ->tuple:
    for i in range(4):
        for j in range(4):
            if pullze[i][j] == 0:
                return (i,j)
def wrongblocks(state:tuple):
    num = 1 #代表了某个位置上正确的数字
    count = 0 #用于统计错位的方块数量
    for i in range(4): #遍历pullze的每一个位置
        for j in range(4):
            if state[i][j] != 0 and state[i][j] != num: #如果该位置不等于0，且不为正确的数字则错位的方块数加一
                count += 1
            num += 1
    return count
def manhatton(state:tuple):
    count = 0 #用于计算所有方块的曼哈顿距离之和
    for i in range(4): #遍历所有方块
        for j in range(4):
            num = state[i][j]
            right_i = (num-1) // 4 #正确的行
            right_j = (num-1) % 4 #正确的列
            if num != 0:
                count += abs(right_i - i) + abs(right_j - j) #计算曼哈顿距离
    return count
def wrongextent(state:tuple):
    count = 0
    num1 = state[0][0]
    for i in range(4):
        for j in range(4):
            if (num1+1) % 16 != state[i][j] and (i != 0 or j != 0):
                count += 1
    return count
def my_heuristic(state:tuple):
    return manhatton(state)+wrongblocks(state)+8*wrongextent(state)
def linearconflict(state:tuple):
    count = 0
    for i in range(4):
        for j in range(4):
            num = state[i][j]
            right_i = (num-1) // 4 
            right_j = (num-1) % 4 
            if num != 0:
                count += abs(right_i - i) + abs(right_j - j) #与曼哈顿启发式相同

                if j == right_j: #如果当前方块在正确的列上
                    for k in range(j+1,4): #遍历在他右边的方块
                        if state[i][k] != 0 and (state[i][k]-1) // 4 == i and (state[i][k]-1) % 4 < j: #存在一个不为0的方块在他右边，但是正确位置在左边
                            count+=2 #该方块至少需要先移开一步再回来，所以至少需要加二
                        break
                if i == right_i: #如果该方块在正确的行上
                    for k in range(i+1,4): #遍历在他下面的方块
                        if state[k][j] != 0 and (state[k][j] -1) // 4 < i and (state[k][i]-1) % 4 == j: #存在一个不为0的方块在他下面，但是正确位置在他上面
                            count+=2 #同理加二
                        break
    return count
"""
函数作用：A*搜索
输入：无
输出：完成pullze的步骤（list），完成每次步骤之后的pullze（list）
以下涉及到的一些变量：
open:PriorityQueue  closed:set
parent:dict  pullze:[[int,int,int,int],[int,int,int,int]...] move_block:(int,int) state:(pullze,move_block)
"""
def A_star():
    global extened_nodes
    open = queue.PriorityQueue() #建立优先队列open表
    closed  = dict() #建立集合closed表
    closed[pullze] = 0 #将初始pullze加入closed表中
    state = (pullze,find_zero_pos(pullze),0) #建立初始状态
    open.put((0,state)) #将f(n)和初始状态加入open表
    parent = {} #用于存下路径
    actions = [] #用于存下每一步的动作
    steps_pullze = [] #用于存下每一步动作之后的pullze
    while(not open.empty()):
        state = open.get()[1] #open优先队列弹出一个状态
        if is_goal(state[0]): #判断是否到达目标状态
            break
        for i in range(4):
            move_block = (state[1][0]+directions[i][0],state[1][1]+directions[i][1]) #计算得到需要移动的方块的位置
            if is_vaild(move_block): #判断该位置是否合理，即坐标不能超过pullze的大小
                new_cost = closed[state[0]]+1 #得到新的g(n)值
                new_pullze = change_to_list(state[0]) #将pullze改为list，因为tuple类型不允许修改，而这里需要交换放块的位置
                new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
                new_pullze = change_to_tuple(new_pullze) #修改回tuple类型，因为list类型不可哈希
                new_state = (new_pullze,move_block,i) #得到新的状态
                if ((not new_pullze in closed) or new_cost < closed[new_pullze]): #如果新状态不在closed中
                    closed[new_pullze] = new_cost
                    open.put((new_cost+my_heuristic(new_pullze),new_state)) #将f(n)和新状态加入open表中
                    parent[new_pullze] = i #记录一下新状态来自的方向
                    extened_nodes += 1 #扩展节点数加一
                    if(extened_nodes % 10000 == 0):
                        print(extened_nodes)
    while(state[0] != pullze): #回溯得到路径，直到回溯到最初输入的pullze
        steps_pullze.append(state[0]) #将每一步完成之后pullze记录下来
        direction = directions[parent[state[0]]]
        move_block = (state[1][0] - direction[0],state[1][1] -direction[1])
        new_pullze = change_to_list(state[0])
        actions.append(new_pullze[move_block[0]][move_block[1]]) #将每一步的动作记录下来
        new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
        new_pullze = change_to_tuple(new_pullze)
        state = (new_pullze,move_block)
    actions.reverse() #reverse记录的动作，使得动作正序输出
    steps_pullze.reverse()
    return actions,steps_pullze
def dfs(limit:int,actions:list,steps_pullze): #深度优先搜索
    global flag
    global goal_state
    global new_limit
    global extened_nodes
    state = steps_pullze[-1] #取上次递归的最后一个状态，主要作用为更新阈值后能接着上次继续深度优先搜索
    if (flag == True): #如果已经找到目标状态就直接退出函数
        return actions,steps_pullze
    if(state[0] == goal): #判断是否找到目标状态
        goal_state = state
        flag =  True
        #print(closed[state[0]]+manhatton(state[0]))
        return actions,steps_pullze
    for i in range(4): #遍历上下左右四个方向
        move_block = (state[1][0]+directions[i][0],state[1][1]+directions[i][1]) #通过计算值为0的方块的某个方向，得到需要移动方块的坐标
        if is_vaild(move_block): #如果该方块是有效的，即该方块的坐标大小不超过pullze大小
            new_cost = closed[state[0]]+1 #新状态的g(n)加一
            new_pullze = change_to_list(state[0])
            new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
            new_pullze = change_to_tuple(new_pullze)
            new_state = (new_pullze,move_block) #得到新状态
            if(new_cost+linearconflict(new_pullze) <= limit and ((not new_pullze in closed) or new_cost<closed[state[0]])):
                closed[new_pullze] = new_cost #将新pullze加入closed
                parent[new_state] = i
                extened_nodes += 1
                actions.append(state[0][move_block[0]][move_block[1]]) #记录步骤
                steps_pullze.append(new_state) #记录每次完成步骤后的pullze
                actions,steps_pullze = dfs(limit,actions,steps_pullze)
                if flag == False: #如果未找到目标状态，则需要弹出已经记录的步骤和pullze等
                    actions.pop()
                    steps_pullze.pop()
                    del closed[new_pullze]
                else:
                    return actions,steps_pullze #否则，直接退出并返回步骤和pullze
            elif(new_cost + linearconflict(new_pullze) > limit): #记录新的阈值
                if new_cost+linearconflict(new_pullze) < new_limit:
                    new_limit = new_cost+linearconflict(new_pullze)
    return actions,steps_pullze
                
def IDA_star():
    global closed
    global parent
    global extened_nodes
    global new_limit
    actions = [] #用于记录每一步的动作
    steps_pullze = [] #用于记录每一步动作之后的pullze
    state = (pullze,find_zero_pos(pullze)) #建立初始状态
    steps_pullze.append(state)   
    closed[pullze] = 0
    while(flag == False): #如果未找到目标状态，则需要更新阈值
        limit = new_limit
        new_limit = 10e9
        actions,steps_pullze = dfs(limit,actions,steps_pullze)
    return actions,steps_pullze #返回步骤和pullze
def display_pullze(pullze:tuple):
    for i in range(4):
        for j in range(4):
            if(pullze[i][j] < 10):
                print(pullze[i][j],end="  ")
            else:
                print(pullze[i][j],end=" ")
        print()
print("input the mode:")
print("A. A star   B. IDA star")
mode = input()
if mode == "A":
    start_time = time.time()
    actions,steps_pullze = A_star()
    end_time = time.time()
elif mode == "B":
    start_time = time.time()
    actions,steps_pullze = IDA_star()
    end_time = time.time()
else:
    print("please input A or B")
    start_time = 0
    end_time = 0
for i in range(0,len(steps_pullze)):
    print("{:-^20}".format("step"+str(i+1)))
    if mode == "A":
        display_pullze(steps_pullze[i])
    elif mode == "B":
        display_pullze(steps_pullze[i][0])
print("Used time: {}s".format(end_time-start_time))
print("steps: {} \n total steps: {} \n extended nodes: {}".format(actions,len(actions),extened_nodes))
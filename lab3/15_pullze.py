import queue
import time
pullze = []
goal = []
directions = [(1,0),(-1,0),(0,1),(0,-1)]
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
def wrongblocks(state:tuple):
    num = 1
    count = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0 and state[i][j] != num:
                count += 1
            num += 1
    return count
def find_zero_pos(pullze:tuple) ->tuple:
    for i in range(4):
        for j in range(4):
            if pullze[i][j] == 0:
                return (i,j)
def manhatton(state:tuple):
    count = 0
    for i in range(4):
        for j in range(4):
            num = state[i][j]
            right_i = (num-1) // 4 
            right_j = (num-1) % 4 
            if num != 0:
                count += abs(right_i - i) + abs(right_j - j)
    return count
extened_nodes = 0
def linearconflict(state:tuple):
    count = 0
    for i in range(4):
        for j in range(4):
            num = state[i][j]
            right_i = (num-1) // 4 
            right_j = (num-1) % 4 
            if num != 0:
                count += abs(right_i - i) + abs(right_j - j)

                if j == right_j:
                    for k in range(j+1,4):
                        if state[i][k] != 0 and (state[i][k]-1) // 4 == i and (state[i][k]-1) % 4 < j:
                            count+=2;
                        break;
                if i == right_i:
                    for k in range(i+1,4):
                        if state[k][j] != 0 and (state[k][j] -1) // 4 < i and (state[k][i]-1) % 4 == j:
                            count+=2;
                        break;
    return count
def A_star():
    global extened_nodes
    open = queue.PriorityQueue()
    closed = set()
    closed.add(pullze)
    state = (pullze,find_zero_pos(pullze),0)
    open.put((0,state))
    parent = {}
    actions = []
    steps_pullze = []
    while(not open.empty()):
        state = open.get()[1]
        if is_goal(state[0]):
            break
        for i in range(4):
            move_block = (state[1][0]+directions[i][0],state[1][1]+directions[i][1])
            if is_vaild(move_block):
                new_cost = state[2]+1
                new_pullze = change_to_list(state[0])
                new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
                new_pullze = change_to_tuple(new_pullze)
                new_state = (new_pullze,move_block,new_cost)
                if is_vaild(move_block) and ((not new_pullze in closed)):
                    closed.add(new_state[0])
                    open.put((new_cost+linearconflict(new_pullze),new_state))
                    parent[new_state] = i
                    extened_nodes += 1
    while(state[0] != pullze):
        steps_pullze.append(state[0])
        old_cost = state[2]-1;
        direction = directions[parent[state]]
        move_block = (state[1][0] - direction[0],state[1][1] -direction[1])
        new_pullze = change_to_list(state[0])
        actions.append(new_pullze[move_block[0]][move_block[1]])
        new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
        new_pullze = change_to_tuple(new_pullze)
        state = (new_pullze,move_block,old_cost)
    actions.reverse()
    steps_pullze.reverse()
    return actions,steps_pullze
closed = {}
parent = {}
new_limit = 0
flag = False
goal_state = ()
def takeFirst(elem):
    return elem[0]
def dfs(state:tuple, limit:int,actions:list,steps_pullze):
    global flag
    global goal_state
    global new_limit
    global extened_nodes
    if (flag == True):
        return actions,steps_pullze
    if(state[0] == goal):
        goal_state = state
        flag =  True
        #print(closed[state[0]]+manhatton(state[0]))
        return actions,steps_pullze
    for i in range(4):
        move_block = (state[1][0]+directions[i][0],state[1][1]+directions[i][1])
        if is_vaild(move_block):
            new_cost = closed[state[0]]+1
            new_pullze = change_to_list(state[0])
            new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
            new_pullze = change_to_tuple(new_pullze)
            new_state = (new_pullze,move_block)
            if(new_cost+manhatton(new_pullze) <= limit and ((not new_pullze in closed) or (new_cost < closed[new_state[0]]))):
                closed[new_pullze] = new_cost
                parent[new_state] = i
                extened_nodes += 1
                actions.append(state[0][move_block[0]][move_block[1]])
                steps_pullze.append(new_pullze)
                actions,steps_pullze = dfs(new_state,limit,actions,steps_pullze)
                if flag == False:
                    actions.pop()
                    steps_pullze.pop()
                else:
                    return actions,steps_pullze
            elif(new_cost + manhatton(new_pullze) > limit):
                if new_cost+manhatton(new_pullze) < new_limit:
                    new_limit = new_cost+manhatton(new_pullze)
    return actions,steps_pullze
                
def IDA_star():
    actions = []
    steps_pullze = []
    global new_limit
    while(flag == False):
        limit = new_limit
        new_limit = 10e9
        global closed
        global parent
        global extened_nodes
        closed[pullze] = 1
        state = (pullze,find_zero_pos(pullze))
        actions,steps_pullze = dfs(state,limit,actions,steps_pullze)
        if( flag == False):
            closed = {}
            parent = {}
            actions = []
            extened_nodes = 0
    return actions,steps_pullze
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
    display_pullze(steps_pullze[i])
print("Used time: {}s".format(end_time-start_time))
print("steps: {} \n total steps: {} \n extended nodes: {}".format(actions,len(actions),extened_nodes))
### 注意new_cost < closed[new_state[0]]
import queue
pullze = []
goal = []
directions = [(1,0),(-1,0),(0,-1),(0,1)]
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
print(row)
print(goal_row)
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
def heuristic(state:tuple):
    num = 1
    count = 0
    for i in range(4):
        for j in range(4):
            if state[i][j] != 0 and state[i][j] != num:
                count += 1
            num += 1
    return count
def heuristic_manhatton(state:tuple):
    count = 0
    for i in range(4):
        for j in range(4):
            num = state[i][j]
            right_i = (num-1) // 4 
            right_j = (num-1) % 4 
            if num != 0:
                count += abs(right_i - i) + abs(right_j - j)
    return count
def a_star():
    frontier = queue.PriorityQueue()
    closed = {}
    closed[pullze] = 1
    state = [pullze,(3,3)]
    frontier.put((0,state))
    while(not frontier.empty()):
        state = frontier.get()[1]
        #state[0] = change_to_tuple(state[0])
        #print(state[1][0],state[1][1])
        if is_goal(state[0]):
            break
        for i in range(4):
            move_block = (state[1][0]+directions[i][0],state[1][1]+directions[i][1])
            if is_vaild(move_block):
                new_cost = closed[state[0]]+1
                new_pullze = change_to_list(state[0])
                new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
                new_pullze = change_to_tuple(new_pullze)
                new_state = [new_pullze,move_block]
                if is_vaild(move_block) and ((not new_pullze in closed) or (new_cost < closed[new_pullze])):
                    #new_pullze = change_to_list(state[0])
                    #new_pullze[move_block[0]][move_block[1]],new_pullze[state[1][0]][state[1][1]] = new_pullze[state[1][0]][state[1][1]],new_pullze[move_block[0]][move_block[1]]
                    #new_state = (new_pullze,move_block)
                    closed[new_pullze] = new_cost
                    frontier.put((new_cost+heuristic_manhatton(new_pullze),new_state))
                    #print(pullze[move_block[0]][move_block[1]],end=" ")


    return 0
a_star()
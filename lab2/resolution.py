import copy
import queue
variables = ['x','y','z','u','v','w','s','t']
functions = ['f','g']
lines = []
success = False
record = []
class Predicate:
    def __init__(self,name):
        self.name = name
        self.variables = []
    def __ne__(self, other: object) -> bool:
        if self.name != other.name:
            return True
        else:
            for i in range(0,len(self.variables)):
                if self.variables[i] != other.variables[i]:
                    return True
        return False

class function:
    def __init__(self,name):
        self.name = name
        self.variables = []

def is_predicate(s):
    for i in range(ord('A'),ord('Z')+1):
        if chr(i) in s:
            return True
    return False

def is_variable(s):
    if s in variables:
        return True
    return False

def is_constant(s):
    if s in variables or s == ",":
        return False
    return True

def delete_repeat_pre(line:list):
    for predicate in line:
        while line.count(predicate) > 2:
            line.remove(predicate)
    return line

def no_repeat_line(add_line:list):
    repeat = True
    for line in lines:
        same = True
        for i in range(0,len(line)):
            if i >= len(add_line):
                same = False
            elif line[i] != add_line[i]:
                same = False
        if same:
            repeat = False
    return repeat

def display_function(f:function):
    res = ""
    res += f.name
    res += "("
    for i in range(0,len(f.variables)):
        if f.variables[i][1] == 2:
            res += display_function(f.variables[i][0])
        else:
            res += f.variables[i][0]
        if i < len(f.variables)-1:
            res += ","
        else:
            res += ")"
    return res

def display(entry):
    add_line,i,j,pre,tar,argu_vars,argu_cons,cur_pos = entry
    if argu_vars != [] and argu_cons != []:
        print("R[%d%c,%d%c]("%(i+1,chr(ord('a')+pre),j+1,chr(ord('a')+tar)),end="")
        for index in range(0,len(argu_vars)):
            var=""
            con=""
            if argu_vars[index][1] == 2:
                var = display_function(argu_vars[index][0])
            else:
                var = argu_vars[index][0]
            if argu_cons[index][1] == 2:
                con = display_function(argu_cons[index][0])
            else:
                con = argu_cons[index][0]
            print("%s=%s"%(var,con),end="")
            if index != len(argu_vars)-1:
                print(",",end="")
            else:
                print(")",end="")
    else :
        print("R[%d%c,%d%c]"%(i+1,chr(ord('a')+pre),j+1,chr(ord('a')+tar)),end="")
    print(" = (",end = "")
    for predicate in add_line:
        print("%s("%predicate.name,end="")
        for argu in predicate.variables:
            if argu[1] == 2:
                print(display_function(argu[0]),end="")
            else:
                print("%s"%argu[0],end="")
            if argu != predicate.variables[-1]:
                print(",",end="")
        print(")",end="")
        if predicate != add_line[-1]:
            print(",",end="")
    print(")")

def get_new_index(i,j,add_lines):
    index1 = -1
    index2 = -1
    for m in range(0,len(add_lines)):
        if add_lines[m] == i:
            index1 = m
        if add_lines[m] == j:
            index2 = m;
    return index1,index2

def display_success(info:tuple):
    q = queue.Queue()
    add_lines = []
    visited = []
    q.put(info)
    while  not q.empty():
        info = q.get()
        add_lines.insert(0,info[-1])
        if info[2] >= n and info[2] not in visited:
            q.put(record[info[2]-n])
            visited.append(info[2])
        if info[1] >= n and info[1] not in visited:
            q.put(record[info[1]-n])
            visited.append(info[1])
    for i in range(0,len(add_lines)):
        entry = record[add_lines[i]-n]
        index1,index2 = get_new_index(entry[1],entry[2],add_lines)
        entry = list(entry)
        if index1 != -1:
            entry[1] = index1+n
        if index2 != -1:
            entry[2] = index2+n
        display(entry)

def sub_func(argu_var,argu_con,t):
    for index in range(0,len(t[0].variables)):
        if t[0].variables[index] == argu_var:
            t[0].variables[index] = argu_con
        elif t[0].variables[index][1] == 2:
            t[0].variables[index] = sub_func(argu_var,argu_con,t[0].variables[index])
    return t

def substiution(argu_var,argu_con,add_line):
    #add_line = copy.deepcopy(lines[i][:pre]+lines[i][pre+1:]+lines[j][:tar]+lines[j][tar+1:])
    for predicate in add_line:
        for m in range(0,len(predicate.variables)):
            if predicate.variables[m] == argu_var:
                predicate.variables[m] = argu_con
            elif predicate.variables[m][1] == 2:
                t = sub_func(argu_var,argu_con,predicate.variables[m])
                predicate.variables[m] = t
    return add_line

def find_func_disagreement(f1:tuple,f2:tuple):
     if f1[0].name == f2[0].name:
         for index in range(0,len(f1[0].variables)):
            if f1[0].variables[index][1] == 1:
                 return f1[0].variables[index],f2[0].variables[index]
            elif f2[0].variables[index][1] == 1:
                return f2[0].variables[index],f1[0].variables[index]
            elif f1[0].variables[index][1] == 2 and f2[0].variables[index][1] == 2:
                return find_func_disagreement(f1[0].variables[index],f2[0].variables[index])

def MGU(predicate,target,i,j,pre,tar):
    flag = False
    add_line = []
    argu_cons = []
    argu_vars = []
    add_line = copy.deepcopy(lines[i][:pre]+lines[i][pre+1:]+lines[j][:tar]+lines[j][tar+1:])
    for m in range(0,len(predicate.variables)):
        if predicate.variables[m] != target.variables[m]:
            if predicate.variables[m][1] == 1 and target.variables[m][1] == 0:
                argu_var = predicate.variables[m]
                argu_con = target.variables[m]
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True
            elif predicate.variables[m][1] == 0 and target.variables[m][1] == 1:
                argu_var = target.variables[m]
                argu_con = predicate.variables[m]
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True
            elif predicate.variables[m][1] == 1 and target.variables[m][1] == 1:
                argu_var = target.variables[m]
                argu_con = predicate.variables[m]
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True
            elif predicate.variables[m][1] == 2 and target.variables[m][1] == 1:
                argu_var = target.variables[m]
                argu_con = predicate.variables[m]
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True
            elif predicate.variables[m][1] == 1 and target.variables[m][1] == 2:
                argu_var = predicate.variables[m]
                argu_con = target.variables[m]
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True
            elif predicate.variables[m][1] == 2 and target.variables[m][1] == 2:
                argu_var,argu_con = find_func_disagreement(predicate.variables[m],target.variables[m])
                argu_vars.append(argu_var)
                argu_cons.append(argu_con)
                flag = True 
            else:
                return False
            if flag == True:
                add_line = substiution(argu_var,argu_con,add_line)
    if no_repeat_line(add_line):
        lines.append(add_line)
        add_line = delete_repeat_pre(add_line)
        cur_pos = len(lines) - 1
        record.append((add_line,i,j,pre,tar,argu_vars,argu_cons,cur_pos))
        #display(add_line,i,j,pre,tar,argu_var,argu_con)
    if add_line == []:
        cur_pos = len(lines)-1
        display_success((add_line,i,j,pre,tar,argu_vars,argu_cons,cur_pos))
        global success
        success = True
    """if flag == False:
        substiution(None,None,i,j,pre,tar,add_line)"""
    return True


def find_clashing_predicate(predicate,target):
    if "¬"+predicate.name == target.name or predicate.name == "¬"+target.name:
        return True
    return False

def resolution(lines):
    length = len(lines)
    i = 0
    while i < length:    
        for j in range(0,i):
            for m in range(0,len(lines[i])):
                for n in range(0,len(lines[j])):
                    if find_clashing_predicate(lines[i][m],lines[j][n]):
                        MGU(lines[i][m],lines[j][n],j,i,n,m)
                        length = len(lines)
                        if success:
                            return True
        i += 1


def is_function(word:str):
    if word in functions:
        return True
    return False

def find_deepest_func(f:function,func_deepth:int,t:tuple):
    if func_deepth == 1:
        return (f.variables.append(t),2)
    else:
        return (find_deepest_func(f.variables[-1][0],func_deepth-1,t))

n = int(input())
for i in range(n):
    s = input()
    predicates = []
    word_list = s.replace('(', ' ( ').replace(')', ' ) ').replace(',',' , ').split()
    #print(word_list)
    for j in range(0,len(word_list)):
        if(is_predicate(word_list[j])):
            p = Predicate(word_list[j])
            #argu_list = word_list[j+2].split(', ')
            create_function = 0
            for index in range(j+2,len(word_list)):
                if is_function(word_list[index]):
                    if create_function == 0:
                        p.variables.append((function(word_list[index]),2))
                        create_function = 1
                    elif create_function > 0:
                        (find_deepest_func(p.variables[-1][0],create_function,(function(word_list[index]),2)))
                        create_function = 2
                elif(word_list[index] == ")"):
                    if create_function == 0:
                        break
                    else:
                        create_function -= 1
                elif(word_list[index] == "," or word_list[index] == "("):
                    continue
                elif is_variable(word_list[index]):
                    if create_function > 0:
                        (find_deepest_func(p.variables[-1][0],create_function,(word_list[index],1)))
                    else:
                        p.variables.append((word_list[index],1))
                else:
                    if create_function > 0:
                        (find_deepest_func(p.variables[-1][0],create_function,(word_list[index],0)))
                
                    else:
                        p.variables.append((word_list[index],0))
            predicates.append(p)
    lines.append(predicates)
resolution(lines)
                    
        



            
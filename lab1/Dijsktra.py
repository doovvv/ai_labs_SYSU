inf = 0x3f3f3f3f
def dijsktra(edges,start,inf):
    distance = {}
    pre = {}
    visited = {}
    for key in edges:
        distance[key] = inf
        visited[key] = False
    distance[start] = 0
    for key in edges:
        index = '!'
        min_num = float('inf')
        for node in edges:
            if (not visited[node]) and distance[node] < min_num:
                min_num = distance[node]
                index = node
        if node == '!':
            break
        visited[index] = True
        for node in edges:
            if (not visited[node]) and index in edges[node] and distance[index]+edges[node][index] < distance[node]:
                distance[node] = distance[index]+edges[node][index]
                pre[node] = index
    return distance,pre
choose = input("choose mode: A.Enter the parameters manually B.Open the file to enter the parameters\nyour choice: ")
first_line = ""
if choose == "B":
    f = open("input.txt")
    l = f.read().split("\n")
    first_line = l[0]
else:
    first_line = input()
nodes_num,edges_num = first_line.split()
nodes_num = int(nodes_num)
edges_num = int(edges_num)
edges = {}
distances = {}
pre = {}
for i in range(edges_num):
    argument = ""
    if choose == "B":
        argument = l[i+1]
    else:
        argument = input()
    node1,node2,distance = argument.split()
    distance = int(distance)
    if node1 not in edges:
        edges[node1] = {}
    edges[node1][node2] = distance
    if node2 not in edges:
        edges[node2] = {}
    edges[node2][node1] = distance
while True:
    target = ""
    if choose == "B":
        target = l[-1]
    else:
        target = input()
    start,end = target.split()
    if start == "exit":
        break
    distances,pre = dijsktra(edges,start,inf)
    print("distance: ",distances[end])
    print("path: %s-->"%end,end = "")
    while(pre[end] != start):
        print("%s-->"%(pre[end]),end="")
        end = pre[end]
    print(start)
    if choose == "B":
        break

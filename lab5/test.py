r=[]
for i in range(4):
    tmp=input().split(' ')
    r+=['(at block%s position%d)'%(x,j+i*4+1) for j,x in enumerate(tmp)]
print(r)
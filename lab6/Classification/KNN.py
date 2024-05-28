import numpy as np
import math
import scipy
#文本输入
def readFile(train:bool):
    if train:
        f = open("train.txt")
    else:
        f = open("test.txt")
    content = f.readlines()
    diff_words = set()
    lines = []
    labels = []
    #print(content)
    for i in range(1,len(content)):
        line = content[i].split()
        temp = []
        labels.append(int(line[1]))
        for j in range(3,len(line)):
            diff_words.add(line[j])
            temp.append(line[j])
        lines.append(temp)
    return lines,np.array(list(diff_words)),np.array(labels)


# 输入句子和所有不同的单词
# 返回特征向量(矩阵)
def boolCountVectorizer(lines,diff_words):
    lines_num = len(lines)
    words_num = len(diff_words)
    mat = np.zeros((lines_num,words_num),dtype=int)
    word_index = {}
    for i in range(words_num):
        word_index[diff_words[i]] = i #提前记录每个单词在单词列表中的下标
    for i in range(lines_num):
        for j in range(len(lines[i])):
            if lines[i][j] in diff_words:
                index = word_index[lines[i][j]]
                mat[i][index] = 1       #将文本中出现的单词在矩阵对应位置设置为1
    return mat


# 输入句子和所有不同的单词
# 返回特征向量(矩阵)
def tf_idfVectorizer(lines,diff_words):
    lines_num = len(lines)
    words_num = len(diff_words)
    mat = np.zeros((lines_num,words_num),dtype=float)
    for i,word in enumerate(diff_words):
        n = 0 # 记录包含word的文档数
        for j in range(lines_num):
            if word in lines[j]:
                n += 1
        idf = math.log((lines_num)/(n+1))
        for j in range(lines_num):
            t = lines[j].count(word)
            tf = t/len(lines[j])   #第j个句子中的tf，即出现频率
            mat[j][i] = tf*idf  #表示第i个词在第j个句子中的tf-idf权重
    return mat


# 输入为特征矩阵，概率，和单词
# 输出为剔除不相关特征后的矩阵，单词
def featureSupport(mat,p,diff_words):
    var = p*(1-p) 
    columns_var = np.var(mat,axis=0) #计算每列的方差
    #print(columns_var)
    mu = columns_var >= var #得到一个全是bool值的向量（False表示被剔除的特征）
    print(mu)
    return mat[:,mu],diff_words[mu] 

# 输入为特征矩阵，相关系数，和单词
# 输出为剔除不相关特征后的矩阵，单词
def featureSupport1(mat,p,diff_words,train_labels):
    lst=[scipy.stats.pearsonr(mat[:,i],train_labels.T)[0] for i in range(mat.shape[1])] #调库计算皮尔逊相关系数
    mu=np.abs(np.array(lst))>=p  #得到一个全是bool值的向量（False表示被剔除的特征）
    return mat[:,mu],diff_words[mu]

def KNN(train_mat,train_labels,test_mat,test_labels,k,p):
    train_num = train_mat.shape[0]
    test_num = test_mat.shape[0]
    pred_true = 0
    for i in range(test_num):
        distances = [np.sum((train_mat[j]-test_mat[i])**p) for j in range(train_num)]
        #distances = [num**(1/p) for num in distances] #省略此步，因为只需要排序，而不需要具体距离
        distances = np.array(distances) #得到距离的向量
        nearest = distances.argsort() #nearest是排序后的下标
        top_k = [train_labels[index] for index in nearest[:k]]  #得到前k个类别
        pred = np.argmax(np.bincount(np.array(top_k))) #topk类别中最大的一个类别
        if pred == test_labels[i]:
            pred_true+=1
    return pred_true

if __name__ == '__main__':
    train_lines,train_diff_words,train_labels = readFile(train=True)
    test_lines,test_diff_words,test_labels = readFile(train=False)
    vectorizer_mode = input("文本向量化方式：\n A.one-hot    B.tf-idf \n你的选择：")
    print()
    if vectorizer_mode == "A":
        train_mat = boolCountVectorizer(train_lines,train_diff_words)
    else:
        train_mat = tf_idfVectorizer(train_lines,train_diff_words)
    fearture_mode = input("特征选择方式：\n A.方差选择法   B.皮尔逊相关系数法   C.不选择\n你的选择：")
    print()
    if fearture_mode == "A":
        p = float(input("输入概率："))
        print()
        train_mat,train_diff_words = featureSupport(train_mat,p,train_diff_words)
    elif fearture_mode == "B":
        p = float(input("输入相关系数："))
        train_mat,train_diff_words = featureSupport1(train_mat,p,train_diff_words,train_labels)
    if vectorizer_mode == "A":
        test_mat = boolCountVectorizer(test_lines,train_diff_words)
    else:
        test_mat = tf_idfVectorizer(test_lines,train_diff_words)
    k,p = eval(input("输入k，p："))
    nums = KNN(train_mat,train_labels,test_mat,test_labels,k,p)
    print()
    print("正确率：%.2f%%"%(nums/test_mat.shape[0]*100))
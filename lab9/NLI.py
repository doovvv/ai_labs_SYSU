import torch
import torch.nn as nn
import numpy as np
import torch.utils
import torch.utils.data
import torchtext;torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import vocab
from torch import optim
from nltk.corpus import stopwords
import random
def get_data(train):
    index = []
    question = []
    sentence = []
    label =[]
    if train:                                                           #区分数据集和训练集
        path = "lab9\\train_40.tsv"
    else:
        path = "lab9\\dev_40.tsv"
    with open(path,encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip('\n').split('\t')                         #按'\t'划分数据
            index.append(line[0])
            question.append(line[1])
            sentence.append(line[2])
            label.append(line[3])
    label_dict = {'entailment':0,'not_entailment':1}
    label = [label_dict[_] for _ in label]                              #将标签转为布尔值
    return index,question,sentence,torch.LongTensor(label).to('cuda')
def load_embedding_matrix():
    embedding_dict = {}
    with open("lab9\\glove.6B.100d.txt",encoding='utf-8') as f:
        for line in f:
            line = line.split()
            word = line[0]
            vector = line[1:]
            embedding_dict[word] = vector
    return embedding_dict
def process_data(lines,vocab_):
    tokenizer  = get_tokenizer('basic_english')             #分词器
    stop_words = set(stopwords.words('english'))            #停用词集合
    lines = [line.replace(',','').replace('?','').replace('.','').replace(':','') for line in lines] #删除标点符号
    processed_lines = [torch.LongTensor([vocab_[word.lower()] for word in tokenizer(line) if (not word.lower() in stop_words) and word.isalnum()]) for line in lines] #分词并删除停用词
    return processed_lines
class QNLIDataset(torch.utils.data.Dataset):
    def __init__(self,dataset) -> None:
        super().__init__()
        self.premise = dataset[0]
        self.hypothese = dataset[1]
        self.label = dataset[2]
    def __getitem__(self, index):
        return (self.premise[index],self.hypothese[index]),self.label[index]
    def __len__(self):
        return len(self.premise)
        

class NLI(nn.Module):
    def __init__(self,embedding_matrix,embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        self.lstm = nn.LSTM(embedding_dim,hidden_size=hidden_dim,num_layers=2,batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*4,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,128)
        self.fc3 = nn.Linear(128,output_dim)
        self.dropout = nn.Dropout(0.2)
    def forward(self,premise,hypothesis):
        embedded_premise = self.embedding(premise)                                  # 前提词嵌入
        embedded_hypothesis = self.embedding(hypothesis)                            # 假设词嵌入
        # LSTM
        _, (hidden_premise, _) = self.lstm(embedded_premise)                        #最终时间步的隐藏状态
        _, (hidden_hypothesis, _) = self.lstm(embedded_hypothesis)

        hidden_premise = torch.cat((hidden_premise[-2,:,:], hidden_premise[-1,:,:]), dim=1)         #拼接前向和后向的隐藏状态
        hidden_hypothesis = torch.cat((hidden_hypothesis[-2,:,:], hidden_hypothesis[-1,:,:]), dim=1)
        
        combined = torch.cat((hidden_premise, hidden_hypothesis), dim=1)            #拼接前提和假设
        combined = self.dropout(nn.functional.relu(self.fc1(combined)))             # 全连接
        combined = self.dropout(nn.functional.relu(self.fc2(combined)))             # 全连接
        return self.fc3(combined)



def categorical_accuracy(preds, y):
    max_preds = preds.argmax(dim=1, keepdim=True)
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() 

# 训练函数
def train(model,epochs,iterator,batch_size,learning_rate):
        # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to('cuda')
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_acc = 0
        model.train()
        sum = 0
        for batch,label in iterator:
            optimizer.zero_grad()
            predictions = model(batch[0].to('cuda'), batch[1].to('cuda'))
            predictions = torch.FloatTensor(predictions.to('cpu')).to('cuda')
            loss = criterion(predictions, label)
            acc = categorical_accuracy(predictions, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss = loss.item()
            epoch_acc = acc.item()
            sum += batch_size

            print("[%d:%d]---准确率：%.2f%%\tloss：%.4f"%(epoch+1,sum,epoch_acc/label.shape[0]*100,epoch_loss))
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)
def test(model,iterator,batch_size):
    model.eval()
    accuarcy = 0
    total = 0
    for batch,label in iterator:
        predictions = model(batch[0].to('cuda'),batch[1].to('cuda'))
        acc = categorical_accuracy(predictions, label)
        accuarcy += acc
        total += label.shape[0]
    print("QNLI准确率：%.2f%%"%(accuarcy/total*100))
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
if __name__== '__main__':
    #[超参数]
    learning_rate = 0.005
    batch_size = 512
    epochs = 5
    hidden_dim = 256
    output_dim = 2
    setup_seed(10)
    #[词字典]
    embedding_dim = 100                             #词向量维度
    embedding_vector = load_embedding_matrix()      #词嵌入字典（单词：向量）
    vocab_ = {word:i+1 for i,word in enumerate(embedding_vector.keys())}            #词嵌入字典（单词：序号）
    vocab_ = vocab(vocab_)
    vocab_.set_default_index(vocab_["<unk>"])       #设置默认下标，碰到未出现的单词自动指向序号0
    #[训练数据的读取和预处理]
    _,premise,hypothese,label = get_data(train=True)#得到测试集数据
    premise = process_data(premise,vocab_)                 #得到预处理之后的前提
    hypothese = process_data(hypothese,vocab_)             #得到预处理之后的假设
    pad_datas_premise = pad_sequence(premise,batch_first=True,padding_value=0)      #将所有前提对齐
    pad_datas_hypothese = pad_sequence(hypothese,batch_first=True,padding_value=0)  #将所有假设对齐
    train_dataset = QNLIDataset((pad_datas_premise,pad_datas_hypothese,label))      #训练数据集
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    #[测试数据的读取和预处理]
    _,premise_test,hypothese_test,label_test = get_data(train=False)#得到测试集数据
    premise_test = process_data(premise_test,vocab_)                 #得到预处理之后的前提
    hypothese_test = process_data(hypothese_test,vocab_)             #得到预处理之后的假设
    pad_datas_premise_test = pad_sequence(premise_test,batch_first=True,padding_value=0)      #将所有前提对齐
    pad_datas_hypothese_test = pad_sequence(hypothese_test,batch_first=True,padding_value=0)  #将所有假设对齐
    test_dataset = QNLIDataset((pad_datas_premise_test,pad_datas_hypothese_test,label_test))      #训练数据集
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    embedding_matrix = torch.FloatTensor(np.zeros((len(vocab_)+1,embedding_dim)))   #初始化嵌入矩阵，每一行为一个词向量
    for i,vector in enumerate(embedding_vector.values()):
        embedding_matrix[i+1] = torch.FloatTensor(np.array(vector,dtype='float32')) #一个列表，元素为词向量

    model = NLI(embedding_matrix,embedding_dim,hidden_dim=hidden_dim,output_dim=output_dim)
    model.to('cuda')
    print("--------train begin--------")
    train(model,epochs,train_dataloader,batch_size,learning_rate)
    print("--------test begin--------")
    test(model,test_dataloader,batch_size)




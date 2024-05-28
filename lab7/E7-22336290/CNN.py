import torch
import torch.utils
import torch.utils.data
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
import time
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def data_process(batch_size):
    #训练时做数据增强
    train_transform = transforms.Compose([transforms.Resize((320,320)), #固定图片尺寸
                                          transforms.RandomHorizontalFlip(),# 随机将图片水平翻转
                                          transforms.RandomRotation(15),# 随机旋转图片 (-15,15)
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    #测试时不需要数据增强
    test_transform = transforms.Compose([transforms.Resize((320,320)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    train_data = torchvision.datasets.ImageFolder("train",transform=train_transform) # 加载数据集
    test_data = torchvision.datasets.ImageFolder("test",transform=test_transform)
    train_data_loader = torch.utils.data.DataLoader(dataset = train_data,batch_size=batch_size, shuffle=True) #划分为mini-batch
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size)
    return train_data_loader,test_data_loader
class CNNnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 模型各层定义
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size= 2)
        self.dp = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64*40*40,640)
        self.fc2 = nn.Linear(640,5)
    # 前向传播函数
    def forward(self,x):
        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = self.pool3(nn.functional.relu(self.conv3(x)))
        x = x.view(-1,64*40*40)
        x = nn.functional.relu(self.fc1(x))
        x = self.dp(x)
        x = nn.functional.relu(self.fc2(x))
        return x
def train(train_data_loader,lr,epochs):
    criterion = nn.CrossEntropyLoss() #损失函数
    optimizer = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9) #优化器
    ## 训练模型
    for epoch in range(epochs):
        running_loss = 0.0
        accuracy = 0.0
        # 迭代，批次训练
        for i, data in enumerate(train_data_loader):
            # 获取训练数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 权重参数梯度清零
            optimizer.zero_grad()  
            # 正向传播
            outputs = net(inputs)
            # 计算损失值
            loss = criterion(outputs, labels)
            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            accuracy += (predicted == labels).sum().item()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 损失值累加
            running_loss += loss.item()
            # 每50个mini-batch显示一次损失值
            if i % 50 == 49:
                print('[%d, %d] loss:%.3f accuracy:%.2f%%' % (epoch + 1, i + 1, running_loss / 50,100*accuracy /(50*batch_size)))
                accuracy_list.append(100*accuracy /(50*batch_size))
                loss_list.append(running_loss / 50)
                running_loss = 0.0
                accuracy = 0.0
    torch.save(net.state_dict(),"CNN.pt")
    print('Finished Training')
def test(test_data_loader):
    net.eval()
    correct = 0    # 预测正确数目
    total = 0     # 测试样本总数
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取批次预测结果
            total += labels.size(0)                    # 批次数目累加
            correct += (predicted == labels).sum().item()  # 预测正确数累加

    print('Accuracy of the network on the %d test images: %d %%' %(total,100*correct/total))
if __name__ == "__main__":
    # 设置参数
    seed =  10
    batch_size = 4
    lr = 0.001
    epoch = 10
    device = 'cuda'
    #固定种子
    setup_seed(seed)
    #创建网络模型
    net  = CNNnet()
    net.to(device=device)
    #模型训练和测试
    loss_list = []
    accuracy_list = []
    mode = input("A.训练模型 B.加载模型\n你的选择：")
    train_data_loader,test_data_loader = data_process(batch_size)
    if mode == "A":
        start_time = time.time()
        train(train_data_loader,lr,epoch)
        end_time = time.time()
        print("total train time %.2fs"%(end_time-start_time))
        test(test_data_loader)
        # 可视化训练结果
        loss_list = np.array(loss_list)
        accuracy_list = np.array(accuracy_list)
        plt.subplot(121),plt.plot(loss_list),plt.title("loss curve")
        plt.subplot(122),plt.plot(accuracy_list),plt.title("accuracy curve")
        plt.show()
    elif mode == "B":
        net.load_state_dict(torch.load("CNN.pt"))
        test(test_data_loader)

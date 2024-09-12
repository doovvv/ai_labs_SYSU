# requirements
# - Python >= 3.7
# - torch >= 1.7
# - gym == 0.23
# - (Optional) tensorboard, wandb

import gym
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
import collections
import random
import math
import matplotlib.pyplot as plt

#神经网络
class QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) #第一层全连接层
        self.fc2 = nn.Linear(hidden_size,hidden_size) #第二层全连接层
        self.fc3 = nn.Linear(hidden_size,output_size) #第三层全连接层
    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.store = collections.deque(maxlen=capacity) #队列模拟经验池

    def len(self):
        return len(self.store) #返回经验池当前大小

    def push(self, *transition):
        self.store.append(transition) #存下一次transition
    def sample(self, batch_size): #随机采样一个batch
        transitions = random.sample(self.store,batch_size) # list,长度为batch_size
        obs, actions, rewards, next_obs, dones = zip(*transitions) #注意为一个batch
        return obs, actions, rewards, next_obs, dones

    def clean(self):
        self.store.clear() #清除经验池

class DQN:
    def __init__(self, env, input_size, hidden_size, output_size):
        self.env = env
        # network for evaluate
        self.eval_net = QNet(input_size, hidden_size, output_size).to(device=device)
        # target network
        self.target_net = QNet(input_size, hidden_size, output_size).to(device=device)
        self.optim = optim.AdamW(self.eval_net.parameters(), lr=args.lr,amsgrad=True)
        self.eps = args.eps
        self.eps_start =args.eps_max
        self.eps_decay =args.eps_decay
        self.eps_end = args.eps_min
        self.gamma = args.gamma
        self.buffer = ReplayBuffer(args.capacity)
        self.loss_fn = nn.MSELoss()
        self.learn_step = 0
    
    def choose_action(self, obs):
        global steps_done
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * steps_done / self.eps_decay)
        steps_done += 1
        obs = torch.unsqueeze(torch.tensor(obs,device=device),dim=0) #转为tensor类型，并增加一维成[1,4]
        # 利用
        if sample > eps_threshold:
            with torch.no_grad():
                actions_value = self.eval_net(obs)
                action = actions_value.argmax().item()
        # 探索
        else:
            action = np.random.randint(low=0,high=2) #0 or 1
        return action
    def store_transition(self, *transition):
        self.buffer.push(*transition)
        
    def learn(self):
        # [Update Target Network Periodically]
        if self.learn_step % args.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step += 1
        
        # [Sample Data From Experience Replay Buffer]
        obs, actions, rewards, next_obs, dones = self.buffer.sample(args.batch_size)
        actions = torch.LongTensor(actions).view(-1,1).to(device=device)  # to use 'gather' latter
        dones = torch.FloatTensor(dones).view(-1,1).to(device=device)
        rewards = torch.FloatTensor(rewards).view(-1,1).to(device=device)
        obs = torch.FloatTensor(np.array(obs)).to(device=device)
        next_obs = torch.FloatTensor(np.array(next_obs)).to(device=device)
        # [learn]
        q_eval = self.eval_net(obs).gather(1,actions) #预测值
        q_target = self.target_net(next_obs).max(1)[0].view(-1,1) #修改维度为[batch_size,1]
        td_target = rewards+self.gamma*(1-dones)*q_target #目标值
        self.optim.zero_grad()
        loss = self.loss_fn(q_eval,td_target) #计算损失
        loss.backward() #反向传播
        self.optim.step() #更新参数



def main():
    rewards_list = []
    averge_rewards = []
    max_average_rewards = 0
    get_goal_time = 0
    flag = True
    env = gym.make(args.env)
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.n
    rewards_100episodes_queue = collections.deque(maxlen=100)
    agent = DQN(env, o_dim, args.hidden, a_dim)                         # 初始化DQN智能体
    for i_episode in range(args.n_episodes):                            # 开始玩游戏
        obs = env.reset()                                           # 重置环境
        episode_reward = 0                                              # 用于记录整局游戏能获得的reward总和
        done = False
        step_cnt=0
        while not done and step_cnt<500:
            step_cnt+=1
            #env.render()                                                # 渲染当前环境(仅用于可视化)
            action = agent.choose_action(obs)                           # 根据当前观测选择动作
            next_obs, reward, done, info = env.step(action)             # 与环境交互
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
            x, x_dot, theta, theta_dot = next_obs
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            new_r = r1 + r2 # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样 DQN 学习更有效率
            agent.store_transition(obs, action, new_r, next_obs, done) # 存储转移
            episode_reward += reward                                    # 记录当前动作获得的reward
            obs = next_obs
            if agent.buffer.len() >= args.batch_size:
                agent.learn()                                           # 学习以及优化网络
        print(f"Episode: {i_episode}, Reward: {episode_reward}")
        rewards_list.append(episode_reward)
        rewards_100episodes_queue.append(episode_reward)
        avg_reward = sum(rewards_100episodes_queue)/len(rewards_100episodes_queue)
        averge_rewards.append(avg_reward)
        if(avg_reward > max_average_rewards):
            max_average_rewards = avg_reward
        if(flag and avg_reward>475):
            get_goal_time = i_episode
            flag = False
    print("最近百局的平均reward值最高为：%d"%max_average_rewards)
    print("最近百局的reward值首次达到475的时间为：第%d个Episode"%get_goal_time)
    plt.figure(figsize=(12,6))
    rewards_list = np.array(rewards_list)
    averge_rewards = np.array(averge_rewards)        
    plt.subplot(121),plt.plot(rewards_list),plt.xlabel("Episode"),plt.ylabel("Reward")
    plt.subplot(122),plt.plot(averge_rewards),plt.axhline(475),plt.xlabel("Episode"),plt.ylabel("Avg_Reward"),plt.ylim((0,500))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="CartPole-v1",  type=str,   help="environment name")
    parser.add_argument("--lr",             default=2e-3,       type=float, help="learning rate")
    parser.add_argument("--hidden",         default=256,         type=int,   help="dimension of hidden layer")
    parser.add_argument("--n_episodes",     default=500,        type=int,   help="number of episodes")
    parser.add_argument("--gamma",          default=0.70,       type=float, help="discount factor")
    # parser.add_argument("--log_freq",       default=100,        type=int)
    parser.add_argument("--capacity",       default=3000,      type=int,   help="capacity of replay buffer")
    parser.add_argument("--eps",            default=0.8,        type=float, help="epsilon of ε-greedy")
    parser.add_argument("--eps_min",        default=0.05,       type=float)
    parser.add_argument("--eps_max",        default=0.9,       type=float)
    parser.add_argument("--batch_size",     default=256,        type=int)
    parser.add_argument("--eps_decay",      default=1000,      type=float)
    parser.add_argument("--update_target",  default=500,        type=int,   help="frequency to update target network")
    args = parser.parse_args()
    device = torch.device('cuda')
    steps_done = 0
    main()
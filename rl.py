# Q-learning
import numpy as np
import random


class game_set:
    # """action->state(环境搭建)"""
    # 输入action 更新state 获取reward(此处同时判断是否游戏结束)
    # 定义环境(maze),observation(当前坐标)和action(创建即获取维度数据 为了构建Q表使用)
    def __init__(self):
        self.maze = np.array([[0, 0, 1, 0, 0, ],
                              [1, 0, 1, 0, 1],
                              [0, 0, 0, 0, 0, ],
                              [0, 0, 1, 1, 0],
                              [0, 0, 0, 0, 2]])
        self.n_rows, self.n_cols = self.maze.shape
        self.actions = ["up", "down", "left", "right"]  # 这里行为表为抽象层 具体层还是数据 本抽象层只有比较作用
        self.n_actions = len(self.actions)
        self.state = np.array([0, 0])
        self.gameon = 1
    def reset(self):
        self.state = np.array([0, 0])
        self.gameon = 1

    # 这里引入语法点 类的所有变量初始化必须显式的调用__init__(self) 只有其内部才可以使用self 如果不使用则成为局部变量 不可被其他成员方法调用
    # 定义环境和行为的交互
    def take_action(self, action):
        if action == "down":
            self.state[0] += 1
        if action == "up":
            self.state[0] -= 1
        if action == "right":
            self.state[1] += 1
        if action == "left":
            self.state[1] -= 1

    def get_next_state(self, action):
        next_state = self.state
        if action == "down":
            next_state[0] += 1
        if action == "up":
            next_state[0] -= 1
        if action == "right":
            next_state[1] += 1
        if action == "left":
            next_state[1] -= 1
        return next_state
    # 定义结束标志
    def check_terminal(self):
        if (self.state[0] not in range(self.n_rows)) or (self.state[1] not in range(self.n_cols)) or self.maze[
            self.state[0], self.state[1]] == 2:
            self.gameon = 0
            return 1
        else:
            return 0

    # """"state->reward层(环境变量获取reward)"""
    # 定义reward
    def get_reward(self):
        if self.check_terminal():
            return -2
        if self.maze[self.state[0], self.state[1]] == 0:
            return -1
        if self.maze[self.state[0], self.state[1]] == 1:
            return -2
        if self.maze[self.state[0], self.state[1]] == 2:
            return 10


# """reward->criterion层(Q表)"""
class Q_learning_method:
    # 输入state 得到action
    def __init__(self):
        self.game_set = game_set()
        self.Q_table = np.zeros((self.game_set.n_rows, self.game_set.n_cols, self.game_set.n_actions), dtype=np.float32)
        self.optim_action = ""
        self.optim_action_index = 0
        self.current_reward = 0
        self.gamma = 0.9
        self.alpha = 0.1
    def check_optim_action(self):
        self.optim_action_index = np.argmax(self.Q_table[self.game_set.state[0], self.game_set.state[1]])
        self.optim_action = self.game_set.actions[self.optim_action_index]

    def simu_annealing_search(self, possibility):
        if random.random() < possibility:
            self.optim_action = self.game_set.actions[random.randint(0, len(self.game_set.actions) - 1)]
        else:
            pass

    def get_max_Qprime(self, next_state):
        return max(self.Q_table[next_state[0], next_state[1]])

    def update_Q_table(self):
        if self.game_set.gameon == 1:
            self.Q_table[self.game_set.state[0], self.game_set.state[1]] += self.alpha * (
                    self.current_reward + self.gamma * self.get_max_Qprime(self.game_set.state) - self.Q_table[
                self.game_set.state[0], self.game_set.state[1]])

    def parameter_update(self, epoch):
        for i in range(epoch):
            self.game_set.reset()  # 每次迭代开始前重置游戏状态
            while self.game_set.gameon == 1:
                self.check_optim_action()
                self.simu_annealing_search(0.3)
                self.game_set.take_action(self.optim_action)
                self.current_reward = self.game_set.get_reward()

            print(self.game_set.gameon)
            print(self.game_set.state)
            # print(self.Q_table)
                # print(self.game_set.state)

Q_learning_method = Q_learning_method()

Q_learning_method.parameter_update(1000)
# 1.为什么print(self.game_set.state)不缩进时不打印---因为gameon一直为零 就没有跳出 不应该在reward处reset
# 2.为什么初始计算的偏好向下---对于argmax对于所有数都相等 默认返回第一个序列
# 3.为什么同一行的值都相等？？？
# import numpy as np
# import random
#
# # 定义迷宫环境 (0: 可走的路, 1: 墙壁, 2: 终点)
# maze = np.array([
#     [0, 0, 1, 0, 0],
#     [1, 0, 1, 0, 1],
#     [0, 0, 0, 0, 0],
#     [0, 0, 1, 1, 0],
#     [0, 0, 0, 0, 2]
# ])
#
# # 状态和动作定义
# n_rows, n_cols = maze.shape
# actions = ['up', 'down', 'left', 'right']
# n_actions = len(actions)
#
# # Q表：记录每个状态下的动作值
# Q_table = np.zeros((n_rows, n_cols, n_actions))#np.zeros输入的是元组
#
# # 超参数
# alpha = 0.1  # 学习率
# gamma = 0.9  # 折扣因子
# epsilon = 0.1  # 探索率
#
#
# # 奖励函数
# def get_reward(state):
#     row, col = state
#     if maze[row, col] == 2:  # 终点
#         return 100
#     elif maze[row, col] == 1:  # 墙壁
#         return -0.2
#     else:  # 每移动一步扣0.1
#         return -0.1
#
#
# # 判断是否到达终点
# def is_terminal(state):
#     return maze[state[0], state[1]] == 2
#
#
# # 定义智能体的移动规则
# def take_action(state, action):
#     row, col = state
#     if action == 'up' and row > 0:
#         row -= 1
#     elif action == 'down' and row < n_rows - 1:
#         row += 1
#     elif action == 'left' and col > 0:
#         col -= 1
#     elif action == 'right' and col < n_cols - 1:
#         col += 1
#     return (row, col)
#
#
# # ε-贪婪策略选择动作
# def choose_action(state, epsilon):
#     if random.uniform(0, 1) < epsilon:
#         return random.randint(0, n_actions - 1)  # 随机选择动作
#     else:
#         row, col = state
#         return np.argmax(Q_table[row, col])  # 选择Q值最大的动作
#
#
# # 训练过程
# n_episodes = 5000  # 训练的回合数
# for episode in range(n_episodes):
#     state = (0, 0)  # 初始状态（左上角）
#
#     while not is_terminal(state):#这就是应该epoch 是否结束游戏
#         # 根据ε-贪婪策略选择动作
#         action_idx = choose_action(state, epsilon)
#         action = actions[action_idx]
#
#         # 执行动作，获取下一个状态
#         next_state = take_action(state, action)
#         reward = get_reward(next_state)
#
#         # Q学习公式更新Q值
#         row, col = state
#         next_row, next_col = next_state
#         Q_table[row, col, action_idx] += alpha * (
#                 reward + gamma * np.max(Q_table[next_row, next_col]) - Q_table[row, col, action_idx]#这里是为了考虑未来的所有r(max Q本身就可以当作一直选择最优解r的递归)
#         )
#
#         # 更新状态
#         state = next_state
#
# # 显示训练后的Q表
# print("训练后的Q表：")
# print(Q_table)
#
#
# # 智能体测试
# def test_agent():
#     state = (0, 0)  # 初始状态
#     path = [state]
#
#     while not is_terminal(state):
#         row, col = state
#         action_idx = np.argmax(Q_table[row, col])  # 选择Q值最大的动作
#         action = actions[action_idx]
#
#         # 执行动作，移动到下一个状态
#         state = take_action(state, action)
#         path.append(state)
#
#     return path
#
#
# # 运行测试，查看智能体找到的路径
# path = test_agent()
# print("智能体找到的路径：")
# print(path)
# #使用Policy Gradient修改训练模型的部分

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
        if action == "up":
            self.state[0] += 1
        if action == "down":
            self.state[0] -= 1
        if action == "right":
            self.state[1] += 1
        if action == "left":
            self.state[1] -= 1

    # 定义结束标志
    def check_terminal(self):
        if (self.state[0] not in range(self.n_rows)) or (self.state[1] not in range(self.n_cols)):
            self.gameon = 0
            return 1
        else:
            return 0

    # """"state->reward层(环境变量获取reward)"""
    # 定义reward
    def get_reward(self):
        if self.check_terminal():
            self.reset()
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

    def check_optim_action(self):
        self.optim_action_index = np.argmax(self.Q_table[self.game_set.state[0], self.game_set.state[1]])
        self.optim_action = self.game_set.actions[self.optim_action_index]

    def simu_annealing_search(self, possibility):
        if random.random() < possibility:
            self.optim_action = self.game_set.actions[random.randint(0, len(self.game_set.actions) - 1)]
        else:
            pass

    def parameter_update(self, epoch):
        for i in range(epoch):
            self.game_set.reset()  # 每次迭代开始前重置游戏状态
            while self.game_set.gameon == 1:
                self.check_optim_action()
                self.simu_annealing_search(0.1)
                self.game_set.take_action(self.optim_action)
                self.current_reward = self.game_set.get_reward()
                # if i in range(0,epoch,100):
                print(self.game_set.state)


Q_learning_method = Q_learning_method()

Q_learning_method.parameter_update(1000)

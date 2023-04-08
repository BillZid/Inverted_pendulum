import pickle
import numpy as np


class MyPendEnv():
    def __init__(self, dis_size=(300, 600)):
        self.m = 0.055
        self.g = 9.81
        self.l_s = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5
        self.Ts = 0.005
        # self.gamma = 0.98
        self.alpha_max = np.pi
        self.alpha_min = -np.pi
        self.alpha_dot_max = 15 * np.pi
        self.alpha_dot_min = -15 * np.pi
        # 离散的动作空间子集，取[-3, 0, 3]个点
        self.action_set = [-3, 0, 3]
        # 离散的状态空间子集，取（300, 600)个点
        self.alpha_set = np.linspace(
            self.alpha_min, self.alpha_max, dis_size[0])
        self.alpha_dot_set = np.linspace(
            self.alpha_dot_min, self.alpha_dot_max, dis_size[1])

    def state_transition(self, state, action):
        # 定义状态转移函数
        alpha, alpha_dot = state
        alpha_dot_dot = (self.m * self.g * self.l_s * np.sin(alpha) - self.b *
                         alpha_dot - np.power(self.K, 2) * alpha_dot/self.R +
                         self.K * action/self.R) / self.J
        alpha_next = alpha + alpha_dot * self.Ts
        alpha_dot_next = alpha_dot + alpha_dot_dot * self.Ts
        # 找到离他们最近的点
        alpha_next = self.alpha_set[np.argmin(
            np.abs(self.alpha_set - alpha_next))]
        alpha_dot_next = self.alpha_dot_set[np.argmin(
            np.abs(self.alpha_dot_set - alpha_dot_next))]
        return np.array([alpha_next, alpha_dot_next])

    def reward(self, state, action):
        # 定义奖励函数
        # 把alpha, alpha_dot, action都扩充为(300, 600, 3)的矩阵
        alpha, alpha_dot = state
        return - 5 * alpha ** 2 - 0.1 * alpha_dot - action ** 2

    def reset(self):
        # 重置环境，返回初始状态
        return np.array([0, 0])

    # def step(self, state, action):
    #     # 环境的一步，返回下一个状态和奖励
    #     state_next = self.state_transition(state, action, self.Ts)
    #     reward = self.reward(state_next, action)
    #     return state_next, reward


class QIteration():
    def __init__(self, env, gamma=0.98, epsilon=0.01):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_function = np.zeros((300, 600, 3))
        self.epoch = 0

    def q_iteration(self):
        # Q迭代算法
        q_function_origin = np.ones((300, 600, 3))
        # 差的范数大于epsilon时继续迭代
        while np.linalg.norm(q_function_origin - self.q_function) > self.epsilon:
            self.epoch += 1
            q_function_origin = self.q_function.copy()
            for i in range(300):
                for j in range(600):
                    for k in range(3):
                        state = np.array([self.env.alpha_set[i],
                                          self.env.alpha_dot_set[j]])
                        action = self.env.action_set[k]
                        state_next = self.env.state_transition(state, action)
                        reward = self.env.reward(state, action)
                        self.q_function[i, j, k] = reward + self.gamma * np.max(
                            self.q_function[np.argmin(
                                np.abs(self.env.alpha_set - state_next[0])),
                                np.argmin(np.abs(
                                    self.env.alpha_dot_set - state_next[1])),
                                :])
            print('epoch: ', self.epoch, 'epsilon: ', np.linalg.norm(
                q_function_origin - self.q_function))

    def get_policy(self):
        # 根据Q函数得到策略
        policy = np.zeros((300, 600))
        for i in range(300):
            for j in range(600):
                policy[i, j] = self.env.action_set[np.argmax(
                    self.q_function[i, j, :])]
        return policy


if __name__ == '__main__':
    env = MyPendEnv()
    q_iteration = QIteration(env)
    q_iteration.q_iteration()
    policy = q_iteration.get_policy()
    # 把policy按行打印出来
    for i in range(300):
        print(policy[i, :])
    # 把模型保存下来
    np.save('./model/policy.npy', policy)
    np.save('./model/q_function.npy', q_iteration.q_function)
    # save the model to disk
    filename = './model/finalized_model.sav'
    pickle.dump(q_iteration, open(filename, 'wb'))

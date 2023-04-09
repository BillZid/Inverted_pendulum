"""
离散化Q迭代算法
"""

from utils import DiscreteUtils
import numpy as np


class DiscreteQIteration(DiscreteUtils):

    def train(self, epsilon=0.01):
        epoch = 0
        # 迭代更新Q表
        while True:
            epoch += 1
            delta = 0
            for i in range(self.n_th):
                for j in range(self.n_thdot):
                    for k in self.env.action_space.n:
                        # 计算Q表
                        th = self.i2v(
                            i, -self.env.th_max, self.env.th_max, self.n_th)
                        thdot = self.i2v(
                            j, -self.env.thdot_max, self.env.thdot_max, self.n_thdot)
                        action = k
                        # 与环境交互
                        u = action * 3 - 3
                        reward = self.env.reward(th, thdot, u)
                        next_th, next_thdot = self.env.new_state(
                            th, thdot, u)
                        # 更新Q表
                        q = reward + self.gamma * \
                            np.max(self.q_table[self.v2i(
                                next_th, -self.env.th_max, self.env.th_max, self.n_th),
                                self.v2i(
                                    next_thdot, -self.env.thdot_max, self.env.thdot_max, self.n_thdot)])
                        delta = max(delta, np.abs(q - self.q_table[i, j, k]))
                        self.q_table[i, j, k] = q
            print("epoch: %i delta: %f" % (epoch, delta))
            if delta < epsilon:
                break

    # 给定状态下，返回最优action
    def do_action(self, th, thdot):
        index_th = self.v2i(
            th, -self.env.th_max, self.env.th_max, self.n_th)
        index_thdot = self.v2i(
            thdot, -self.env.thdot_max, self.env.thdot_max, self.n_thdot)
        action = np.argmax(self.q_table[index_th, index_thdot])
        return action


if __name__ == "__main__":

    agent = DiscreteQIteration(
        n_th=600, n_thdot=600, n_actions=7, gamma=0.98)
    agent.load()  # load参数
    # agent.train(epsilon=0.01)
    # agent.save()
    agent.demo(max_step=500,save_video=True)  # 保存视频
    agent.env.close()

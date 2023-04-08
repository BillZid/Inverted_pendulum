"""
离散化Q迭代算法
"""

from base_agent import BaseAgent
import numpy as np
import matplotlib


class DiscreteQIteration(BaseAgent):

    def train(self, epsilon=0.01):
        epoch = 0
        # 迭代更新Q表
        while True:
            epoch += 1
            delta = 0
            for i in range(self.n_th):
                for j in range(self.n_thdot):
                    for k in range(len(self.actions)):
                        # 计算Q表
                        th = self.index_to_value(
                            i, -self.th_max, self.th_max, self.n_th)
                        thdot = self.index_to_value(
                            j, -self.thdot_max, self.thdot_max, self.n_thdot)
                        action = self.actions[k]
                        # 与环境交互
                        reward = self.env.reward(th, thdot, action)
                        next_th, next_thdot = self.env.new_state(
                            th, thdot, action)
                        # 更新Q表
                        q = reward + self.gamma * \
                            np.max(self.q_table[self.value_to_index(
                                next_th, -self.th_max, self.th_max, self.n_th),
                                self.value_to_index(
                                    next_thdot, -self.thdot_max, self.thdot_max, self.n_thdot)])
                        delta = max(delta, np.abs(q - self.q_table[i, j, k]))
                        self.q_table[i, j, k] = q
            print("epoch: %i delta: %f" % (epoch, delta))
            if delta < epsilon:
                break

    # 给定状态下，返回最优action
    def do_action(self, th, thdot):
        index_th = self.value_to_index(
            th, -self.th_max, self.th_max, self.n_th)
        index_thdot = self.value_to_index(
            thdot, -self.thdot_max, self.thdot_max, self.n_thdot)
        action_index = np.argmax(self.q_table[index_th, index_thdot])
        return np.array([self.actions[action_index]])


if __name__ == "__main__":

    agent = DiscreteQIteration(
        n_th=500, n_thdot=500, n_actions=3, gamma=0.98)
    agent.load()  # load参数
    # agent.train(epsilon=0.01)
    # agent.save()
    agent.demo(save_video=True)  # 保存视频
    agent.env_close()

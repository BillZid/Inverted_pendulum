"""
SARSA算法
"""
import numpy as np
from utils import DiscreteUtils


class SARSA(DiscreteUtils):

    # episode_num: episode数目，episode_length: 一个episode的长度
    def train(self, episode_num=100000, episode_length=10000):

        lr = 0.8
        epsilon = 0.1  # 初始的epsilon贪心的探索率
        # decay = 1  # epsilon和学习率的衰减率
        # epsilon_low = 0.01  # epsilon的下界
        gamma = self.gamma

        th_max = self.env.th_max
        thdot_max = self.env.thdot_max

        episode_count = 0  # 采样了多少个episode
        episode_return = 0
        while True:
            episode_count += 1
            (th, thdot), _ = self.env.reset(fix=True)
            i_th = self.v2i(
                    th, -th_max, th_max, self.n_th)
            i_thdot = self.v2i(
                    thdot, -thdot_max, thdot_max, self.n_thdot)
            action = self.epsilon_greedy_action(i_th, i_thdot, epsilon)
            episode_step = 0  # 当前episode采样了多少个点
            delta = 0
            while episode_step < episode_length:
                old_q_table = self.q_table.copy()
                episode_step += 1
                (new_th, new_thdot), reward, ter, _, _ = self.env.step(action)
                episode_return += reward
                i_new_th = self.v2i(
                    new_th, -th_max, th_max, self.n_th)
                i_new_thdot = self.v2i(
                    new_thdot, -thdot_max, thdot_max, self.n_thdot)
                new_action = self.epsilon_greedy_action(
                    i_new_th, i_new_thdot, epsilon)
                self.q_table[i_th, i_thdot, action] += lr * (
                    reward + gamma*self.q_table[i_new_th, i_new_thdot, new_action] - 
                    self.q_table[i_th, i_thdot, action])
                action = new_action
                i_th, i_thdot = i_new_th, i_new_thdot
                if ter:
                    break
                delta = max(delta, np.abs(np.amax(self.q_table - old_q_table)))
            with open(self.data_dir + 'log.txt', 'a+') as f:
                f.write('episode %i: delta= %s return= %s \n' % (episode_count, delta, episode_return))
                episode_return = 0
            if episode_count % 100 == 0:
                np.save(self.data_dir + 'q_table_%i.npy' % (episode_count), self.q_table)
                # lr = lr * decay
                # epsilon = max(epsilon * decay, epsilon_low)

            # if episode_count % 100 == 0:
            #     print("episode %s:  delta= %s" % (episode_count, delta))
            #     print("episode %s:  return= %s" % (episode_count, episode_return))
            #     episode_return = 0

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, index_th, index_thdot, epsilon):
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            action = np.argmax(self.q_table[index_th, index_thdot])
        else:  # 以1-epsilon的概率随机返回一个action
            action = np.random.randint(2)
        return action

    # 给定状态下，返回最优action
    def do_action(self, th, thdot):
        index_th = self.v2i(
            th, -self.env.th_max, self.env.th_max, self.n_th)
        index_thdot = self.v2i(
            thdot, -self.env.thdot_max, self.env.thdot_max, self.n_thdot)
        action = np.argmax(self.q_table[index_th, index_thdot])
        return action


if __name__ == "__main__":
    agent = SARSA(n_th=300, n_thdot=300, n_actions=3, gamma=0.98)
    # agent.load()  # load参数
    agent.train()  # 训练模型
    # agent.save()
    # agent.demo(max_step=1000)  # 演示
    # agent.demo(save_video=True)  # 保存视频
    agent.env.close()

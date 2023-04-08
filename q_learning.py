"""
QLearning算法
"""
import numpy as np
from base_agent import BaseAgent

np.random.seed(0)


class QLearning(BaseAgent):

    # episode_num: episode数目，episode_length: 一个episode的长度
    def train(self, episode_num=500, episode_length=20000):

        lr = 0.1
        decay = 1  # epsilon和学习率的衰减率
        epsilon = 0.1  # 初始的epsilon贪心的探索率
        epsilon_low = 0.01  # epsilon的下界
        gamma = self.gamma

        # diff_list = []
        global_step = 0  # q_table进行的多少次更新
        episode_count = 0  # 采样了多少个episode
        episode_return = 0
        while episode_count < episode_num:
            episode_count += 1
            old_q_table = self.q_table.copy()
            (th, thdot), _ = self.env.reset(fix=True)
            episode_step = 0  # 当前episode采样了多少个点
            while episode_step < episode_length:
                episode_step += 1
                global_step += 1
                action = self.epsilon_greedy_action(th, thdot, epsilon)
                index_th = self.value_to_index(
                    th, -self.th_max, self.th_max, self.n_th)
                index_thdot = self.value_to_index(
                    thdot, -self.thdot_max, self.thdot_max, self.n_thdot)
                index_action = self.value_to_index(
                    action, -self.umax, self.umax, self.n_actions)

                (new_th, new_thdot), reward, terminated, _, _ = self.env.step(action)
                episode_return += reward
                index_new_th = self.value_to_index(
                    new_th, -self.th_max, self.th_max, self.n_th)
                index_new_thdot = self.value_to_index(
                    new_thdot, -self.thdot_max, self.thdot_max, self.n_thdot)
                self.q_table[index_th, index_thdot, index_action] = lr * (
                    reward + gamma*np.amax(self.q_table[index_new_th, index_new_thdot]) - 
                    self.q_table[index_th, index_thdot, index_action]) + self.q_table[index_th, index_thdot, index_action]
                # if terminated:
                #    break
                th, thdot = new_th, new_thdot
                lr = lr * decay
                epsilon = max(epsilon * decay, epsilon_low)
            delta = np.abs(np.amax(self.q_table - old_q_table))
            if episode_count % 100 == 0:
                print("episode %s:  delta= %s" % (episode_count, delta))
                print("episode %s:  return= %s" % (episode_count, episode_return/(100*2000)))
                episode_return = 0

    # 通过epsilon贪心策略返回一个action
    def epsilon_greedy_action(self, th, thdot, epsilon):
        index_th = self.value_to_index(
            th, -self.th_max, self.th_max, self.n_th)
        index_thdot = self.value_to_index(
            thdot, -self.thdot_max, self.thdot_max, self.n_thdot)
        if np.random.rand() > 1-epsilon:  # 以1-epsilon的概率返回q最大的action
            index_action = np.argmax(self.q_table[index_th, index_thdot])
            action = self.actions[index_action]
        else:  # 以1-epsilon的概率随机返回一个action
            action = np.random.choice(self.actions)
        action = action if isinstance(
            action, np.ndarray) else np.array([action])
        return action

    # 给定状态下，返回最优action
    def do_action(self, th, thdot):
        index_th = self.value_to_index(
            th, -self.th_max, self.th_max, self.n_th)
        index_thdot = self.value_to_index(
            thdot, -self.thdot_max, self.thdot_max, self.n_thdot)
        action_index = np.argmax(self.q_table[index_th, index_thdot])
        return np.array([self.actions[action_index]])


if __name__ == "__main__":
    agent = QLearning(n_th=300, n_thdot=300, n_actions=3, gamma=1)
    agent.load()  # load参数
    agent.train()  # 训练模型
    agent.save()
    # agent.demo(max_step=1000)  # 演示
    agent.demo(save_video=True)  # 保存视频
    agent.env_close()

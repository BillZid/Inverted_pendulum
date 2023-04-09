"""
智能体设定，离散化操作
"""
import numpy as np
import os
from gym.wrappers.record_video import RecordVideo
from envs.my_pend_env import MyPendEnv


class DiscreteUtils():
    def __init__(self, n_th=300, n_thdot=300, n_actions=3, gamma=0.98, render_mode="rgb_array"):
        self.model_name = self.__class__.__name__  # 模型名称
        self.gamma = gamma  # 衰减因子
        self.env = MyPendEnv(render_mode=render_mode)

        # 离散化参数
        self.n_th = n_th
        self.n_thdot = n_thdot
        self.n_actions = n_actions

        # q矩阵
        self.q_table = np.zeros((self.n_th, self.n_thdot, self.n_actions))

        self.data_dir = "./results/%s_th-%s_dth-%s_a-%s/" % (
            self.model_name, self.n_th, self.n_thdot, self.n_actions)  # 数据保存路径
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    # 每个算法定义自己的train
    def train(self):
        pass

    # 给定状态下，返回最优action
    def do_action(self, th, thdot):
        pass

    # 把离散区间id（从0开始）映射成区间的值（以中点的值为代表）参数：区间id，总区间起始值，总区间终止值，区间总数
    def i2v(self, index, start, end, numbers):
        one_length = (end-start) / numbers  # 区间的单位长度
        return start + one_length / 2 + index * one_length

    # 把连续值映射成离散区间id。参数：值，总区间起始值，总区间终止值，区间总数
    def v2i(self, value, start, end, numbers):
        value = value if isinstance(value, np.ndarray) else np.array([value])
        index = ((value-start) / (end-start) * numbers).astype(int)  # 将小数向下取整到最接近的整数
        index[index == numbers] -= 1
        return index

    # 可视化
    def demo(self, max_step=1000, save_video=True):
        if save_video:
            self.env = RecordVideo(self.env, self.data_dir + "demo_video")
        observation, _ = self.env.reset(fix=True)
        step = 0
        while step < max_step:
            step += 1
            th, thdot = observation
            action = self.do_action(th, thdot)
            observation, _, _, _, _ = self.env.step(action)

    def save(self):
        np.save(self.data_dir + "checkpoint.npy", self.q_table)

    def load(self):
        if os.path.exists(self.data_dir + "checkpoint.npy"):
            file = self.data_dir + "checkpoint.npy"
            self.q_table = np.load(file)

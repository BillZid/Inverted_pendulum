import numpy as np
from os import path
import gym
from gym import spaces
from typing import Optional
import pygame
from pygame import gfxdraw


class MyPendEnv(gym.Env):
    metadata = {
            "render_modes": ["human", "rgb_array"],
            "render_fps": 30,
        }

    def __init__(self, render_mode):

        # 基础参数设定
        self.m = 0.055
        self.g = 9.81
        self.ls = 0.042
        self.J = 1.91e-4
        self.b = 3e-6
        self.K = 0.0536
        self.R = 9.5
        self.Ts = 0.005
        self.th_max = np.pi
        self.thdot_max = 15 * np.pi
        self.thdot_min = -15 * np.pi
        self.umax = 3
        self.umin = -3

        # render设定
        self.render_mode = render_mode
        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        # 动作空间设定
        # self.action_space = spaces.Box(
        #     low=self.umin, high=self.umax, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        # 状态空间设定
        high = np.array([self.th_max, self.thdot_max], dtype=np.float32)
        low = -high
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32)

    def step(self, action):
        th, thdot = self.state  # th := theta, 角度；thdot：角速度
        u = action * 3 - 3
        self.last_u = u  # for rendering
        reward = self.reward(th, thdot, u)
        newth, newthdot = self.new_state(th, thdot, u)
        self.state = np.array([newth, newthdot])  # 状态更新
        if self.render_mode == "human":
            self.render()
        terminated = np.equal(self.state, np.array([0., 0.])).all()  # 是否达到终止状态
        return self._get_obs(), reward, terminated, False, {}

    def new_state(self, th, thdot, u):
        g = self.g
        m = self.m
        ls = self.ls
        b = self.b
        K = self.K
        R = self.R
        J = self.J
        Ts = self.Ts
        thdotdot = (m * g * ls * np.sin(th) - b * thdot - np.power(K, 2) *
                    thdot/R + K * u/R) / J
        newth = th + thdot * Ts
        newth = np.mod(newth + np.pi, 2 * np.pi) - np.pi
        newthdot = thdot + thdotdot * Ts
        newthdot = np.clip(newthdot, self.thdot_min, self.thdot_max)
        return np.array([newth, newthdot])

    def reset(self, fix=False, seed: Optional[int] = None):
        super().reset(seed=seed)  # 从父类中引入属性,seed是下面uniform的随机种子
        high = np.array([self.th_max, self.thdot_max], dtype=np.float32)
        low = -high  # We enforce symmetric limits.
        if fix:
            self.state = np.array([-np.pi, 0], dtype=np.float32)
        else:
            self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def _get_obs(self):
        th, thdot = self.state
        return np.array([th, thdot], dtype=np.float32)

    def render(self):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_dim, self.screen_dim)
                )
            else:  # mode in "rgb_array"
                self.screen = pygame.Surface((self.screen_dim, self.screen_dim))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((self.screen_dim, self.screen_dim))
        self.surf.fill((255, 255, 255))

        bound = 2.2
        scale = self.screen_dim / (bound * 2)
        offset = self.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.2 * scale
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(self.state[0] + np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(self.surf, transformed_coords, (204, 77, 77))
        gfxdraw.filled_polygon(self.surf, transformed_coords, (204, 77, 77))

        gfxdraw.aacircle(self.surf, offset, offset, int(rod_width / 2), (204, 77, 77))
        gfxdraw.filled_circle(
            self.surf, offset, offset, int(rod_width / 2), (204, 77, 77)
        )

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(self.state[0] + np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )
        gfxdraw.filled_circle(
            self.surf, rod_end[0], rod_end[1], int(rod_width / 2), (204, 77, 77)
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if self.last_u is not None:
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(self.last_u) / 2, scale * np.abs(self.last_u) / 2),
            )
            is_flip = bool(self.last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    offset - scale_img.get_rect().centerx,
                    offset - scale_img.get_rect().centery,
                ),
            )

        # drawing axle
        gfxdraw.aacircle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(self.surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        else:  # mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def reward(self, theta, thdot, u):
        # cost是reward的相反数，是正的。reward是负的
        theta = np.mod(theta + np.pi, 2 * np.pi) - np.pi
        costs = 5 * theta ** 2 + 0.1 * thdot ** 2 + u ** 2
        return -costs


if __name__ == '__main__':
    env = MyPendEnv(render_mode="rgb_array")
    env.action_space.seed(2)
    observation, info = env.reset(fix=True)
    env.render()
    for i in range(1000):
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        # print(observation, reward)
        env.render()
        print(env.last_u)
        if terminated:
            break
    env.close()

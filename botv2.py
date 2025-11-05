

import os
import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from multiprocessing import cpu_count
from typing import Callable, List, Tuple

import distutils.util

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback 



MODEL_FILENAME = "ppo_battlebots_agent_v2.zip"



ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 700.0
ARENA_TOP = 10.0
ARENA_LEFT = 0.0
BOT_RADIUS = 13.0
BOT_DIAMETER = BOT_RADIUS * 2.0
BOT_SPEED = 3.0
BULLET_SPEED = 5.0
MAX_BULLETS_PER_BOT = 4
TIME_LIMIT_SECS = 90
FRAMES_PER_SEC = 30
MAX_STEPS = TIME_LIMIT_SECS * FRAMES_PER_SEC 


ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_FIREUP = 4
ACTION_FIREDOWN = 5
ACTION_FIRELEFT = 6
ACTION_FIRERIGHT = 7
ACTION_STAY = 8



TRAINING_KWARGS = {
    'width': ARENA_WIDTH,
    'height': ARENA_HEIGHT,
    'max_steps': 1000, 
    'min_bots': 10,
    'max_bots': 16,
    'max_bullets_per_bot': MAX_BULLETS_PER_BOT,
    'bot_radius': BOT_RADIUS,
    'bot_speed': BOT_SPEED,
    'bullet_speed': BULLET_SPEED,
    'win_reward': 15.0,            
    'death_penalty': -50.0,        
    'kill_reward': 10.0,            
    'survival_bonus_per_step': 0.5 / FRAMES_PER_SEC, 
    'wall_penalty': -0.2           
}

TESTING_KWARGS = {
    'width': ARENA_WIDTH,
    'height': ARENA_HEIGHT,
    'max_steps': MAX_STEPS,
    'min_bots': 16,
    'max_bots': 16,
    'max_bullets_per_bot': MAX_BULLETS_PER_BOT,
    'bot_radius': BOT_RADIUS,
    'bot_speed': BOT_SPEED,
    'bullet_speed': BULLET_SPEED,
    'win_reward': 15.0,
    'death_penalty': -10.0,
    'kill_reward': 5.0,
    'survival_bonus_per_step': 0.1 / FRAMES_PER_SEC,
    'wall_penalty': -0.2
}


SCREEN_WIDTH = int(ARENA_WIDTH)
SCREEN_HEIGHT = int(ARENA_HEIGHT)
FRAME_RATE = 30 

COLOR_BG = (10, 10, 10); COLOR_WALL = (40, 40, 40); COLOR_AGENT = (50, 150, 255)
COLOR_BOT = (255, 150, 50); COLOR_BULLET_AGENT = (100, 200, 255); COLOR_BULLET_BOT = (255, 150, 150)
COLOR_DEAD = (100, 100, 100)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def str_to_bool(val):
        """Converts a string to a boolean."""
        try:
            return bool(distutils.util.strtobool(val))
        except ValueError:
            return False

class BattleBotsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=1000.0, height=700.0, max_steps=2700, min_bots=1,
                 max_bots=5, max_bullets_per_bot=4, bot_radius=13.0,
                 bot_speed=3.0, bullet_speed=6.0, win_reward=10.0,
                 death_penalty=-10.0, kill_reward=5.0,
                 survival_bonus_per_step=0.0033, wall_penalty=-0.2):
        super().__init__()
        
        self.width = float(width); self.height = float(height)
        self.max_steps = int(max_steps); self.min_bots = int(min_bots)
        self.max_bots = int(max_bots); self.max_bullets_per_bot = int(max_bullets_per_bot)
        self.bot_radius = float(bot_radius); self.bot_diameter = self.bot_radius * 2.0
        self.bot_speed = float(bot_speed); self.bullet_speed = float(bullet_speed)
        
        
        self.win_reward = float(win_reward); self.death_penalty = float(death_penalty)
        self.kill_reward = float(kill_reward)
        self.survival_bonus = float(survival_bonus_per_step)
        self.wall_penalty = float(wall_penalty)
        
        
        self.min_x = ARENA_LEFT
        self.max_x = self.width - self.bot_diameter
        self.min_y = ARENA_TOP
        self.max_y = self.height - self.bot_diameter

        
        self.BOT_TYPE_RAND = 0
        self.BOT_TYPE_HUNTER = 1
        self.BOT_TYPE_SPINNER = 2
        self.BOT_TYPE_WALL_HUGGER = 3

        
        
        self.action_space = spaces.Discrete(9)
        
        self.max_total_bullets = (self.max_bots + 1) * self.max_bullets_per_bot
        
        
        
        
        
        
        
        obs_frame_len = 2 + 1 + (self.max_bots * 3) + (self.max_total_bullets * 2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_frame_len * 2,), dtype=np.float32
        )
        
        
        self.prev_obs = None
        self.current_step = 0
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_ammo = 0
        
        
        
        
        
        
        
        
        
        
        self.other_bots_state = np.zeros((self.max_bots, 8), dtype=np.float32)
        self.num_bots_alive = 0
        
        
        self.bullets = np.zeros((self.max_total_bullets, 5), dtype=np.float32)
        self.num_bullets_active = 0
        
    def _get_obs(self):
        
        agent_pos_norm = np.array([
            (self.agent_pos[0] - self.min_x) / (self.max_x - self.min_x) * 2 - 1,
            (self.agent_pos[1] - self.min_y) / (self.max_y - self.min_y) * 2 - 1
        ], dtype=np.float32)
        
        
        agent_ammo_norm = np.array([(self.agent_ammo / self.max_bullets_per_bot) * 2 - 1], dtype=np.float32)
        
        
        bot_info_padded = np.full((self.max_bots, 3), -1.0, dtype=np.float32)
        if self.num_bots_alive > 0:
            live_bots = self.other_bots_state[:self.num_bots_alive]
            
            bot_pos_norm = np.array([
                (live_bots[:, 1] - self.min_x) / (self.max_x - self.min_x) * 2 - 1,
                (live_bots[:, 2] - self.min_y) / (self.max_y - self.min_y) * 2 - 1
            ]).T
            bot_ammo_norm = (live_bots[:, 3] / self.max_bullets_per_bot) * 2 - 1
            bot_info_padded[:self.num_bots_alive, :2] = bot_pos_norm
            bot_info_padded[:self.num_bots_alive, 2] = bot_ammo_norm
        
        
        bullet_info_padded = np.full((self.max_total_bullets, 2), -1.0, dtype=np.float32)
        if self.num_bullets_active > 0:
            active_bullets = self.bullets[:self.num_bullets_active, :2]
            bullet_pos_norm = np.array([
                (active_bullets[:, 0] - self.min_x) / (self.width - self.min_x) * 2 - 1,
                (active_bullets[:, 1] - self.min_y) / (self.height - self.min_y) * 2 - 1
            ]).T
            bullet_info_padded[:self.num_bullets_active] = bullet_pos_norm

        
        current_obs = np.concatenate([
            agent_pos_norm, 
            agent_ammo_norm, 
            bot_info_padded.flatten(), 
            bullet_info_padded.flatten()
        ]).astype(np.float32)
        
        if self.prev_obs is None: self.prev_obs = current_obs
        stacked_obs = np.concatenate([self.prev_obs, current_obs])
        self.prev_obs = current_obs
        return stacked_obs

    def _add_bullet(self, x, y, dx, dy, owner_id):
        if self.num_bullets_active < self.max_total_bullets:
            self.bullets[self.num_bullets_active] = [x, y, dx, dy, owner_id]
            self.num_bullets_active += 1
            return True
        return False

    def reset(self, seed=None, options=None):
        
        super().reset(seed=seed) 
        self.prev_obs = None
        self.current_step = 0
        self.num_bullets_active = 0
        
        
        self.num_bots_alive = self.np_random.integers(self.min_bots, self.max_bots + 1)
        
        
        positions = []
        spawn_count = self.num_bots_alive + 1 
        x_scale = (self.width - self.bot_diameter * 2) / max(spawn_count - 1, 1)
        y_scale = (self.height - self.bot_diameter * 2) / 5 
        
        occupied_cells = set()
        for i in range(spawn_count):
            while True:
                x_idx = self.np_random.integers(0, spawn_count)
                y_idx = self.np_random.integers(0, 5)
                if (x_idx, y_idx) not in occupied_cells:
                    occupied_cells.add((x_idx, y_idx))
                    x = self.min_x + x_idx * x_scale + self.bot_radius
                    y = self.min_y + y_idx * y_scale + self.bot_radius
                    positions.append([x, y])
                    break
        
        self.np_random.shuffle(positions)
        
        self.agent_pos = np.array(positions.pop(), dtype=np.float32)
        self.agent_ammo = self.max_bullets_per_bot
        
        for i in range(self.num_bots_alive):
            pos = positions.pop()
            bot_type = self.np_random.integers(0, 4) 
            
            self.other_bots_state[i, 0] = bot_type
            self.other_bots_state[i, 1] = pos[0] 
            self.other_bots_state[i, 2] = pos[1] 
            self.other_bots_state[i, 3] = self.max_bullets_per_bot 
            
            if bot_type == self.BOT_TYPE_RAND:
                self.other_bots_state[i, 4] = self.np_random.integers(0, 4) 
                self.other_bots_state[i, 5] = 99 
            elif bot_type == self.BOT_TYPE_HUNTER:
                self.other_bots_state[i, 4] = ACTION_STAY
                self.other_bots_state[i, 5] = 0 
            elif bot_type == self.BOT_TYPE_SPINNER:
                self.other_bots_state[i, 4] = ACTION_FIREUP
                self.other_bots_state[i, 5] = 15 
            elif bot_type == self.BOT_TYPE_WALL_HUGGER:
                self.other_bots_state[i, 4] = ACTION_RIGHT 
                self.other_bots_state[i, 5] = 20 
            
        return self._get_obs(), {}

    def _get_rand_move(self, bot_state):
        """Implements RandBot logic"""
        current_move = int(bot_state[4])
        move_count = bot_state[5]
        
        move_count += 1
        
        
        if move_count >= 30 + self.np_random.integers(0, 60):
            move_count = 0
            choice = self.np_random.integers(0, 8) 
            
            if choice == 0: current_move = ACTION_UP
            elif choice == 1: current_move = ACTION_DOWN
            elif choice == 2: current_move = ACTION_LEFT
            elif choice == 3: current_move = ACTION_RIGHT
            elif choice == 4: current_move = ACTION_FIREUP
            elif choice == 5: current_move = ACTION_FIREDOWN
            elif choice == 6: current_move = ACTION_FIRELEFT
            elif choice == 7: current_move = ACTION_FIRERIGHT
            
            
            if choice >= 4:
                move_count = 99 
        
        bot_state[4] = current_move
        bot_state[5] = move_count
        
        return current_move, bot_state

    def _get_hunter_move(self, bot_idx, bot_state, shot_ok):
        """Implements a smart Hunter Bot"""
        bot_pos = bot_state[1:3]
        bot_center = bot_pos + self.bot_radius
        counter = bot_state[5]
        
        target_pos = bot_state[6:8]
        
        
        counter += 1
        if counter > 20 or np.array_equal(target_pos, [0, 0]):
            counter = 0
            
            
            targets = []
            
            targets.append((self.agent_pos + self.bot_radius, 0))
            
            for i in range(self.num_bots_alive):
                if i != bot_idx:
                    targets.append((self.other_bots_state[i, 1:3] + self.bot_radius, i + 1))
            
            
            min_dist = float('inf')
            best_target = None
            for t_pos, t_id in targets:
                dist = np.linalg.norm(bot_center - t_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target = t_pos
            
            if best_target is not None:
                target_pos = best_target 
            bot_state[6:8] = target_pos

        bot_state[5] = counter
        
        
        dx = target_pos[0] - bot_center[0]
        dy = target_pos[1] - bot_center[1]
        
        alignment_threshold = self.bot_radius * 1.5
        
        
        if shot_ok:
            if abs(dy) < alignment_threshold: 
                if dx > 0: return ACTION_FIRERIGHT, bot_state
                else: return ACTION_FIRELEFT, bot_state
            if abs(dx) < alignment_threshold: 
                if dy > 0: return ACTION_FIREDOWN, bot_state
                else: return ACTION_FIREUP, bot_state
                
        
        if abs(dx) > abs(dy): 
            if dx > 0: return ACTION_RIGHT, bot_state
            else: return ACTION_LEFT, bot_state
        else: 
            if dy > 0: return ACTION_DOWN, bot_state
            else: return ACTION_UP, bot_state

    def _get_spinner_move(self, bot_state):
        """Implements a stationary spinner bot"""
        current_move = int(bot_state[4])
        counter = bot_state[5]
        
        counter -= 1
        
        if counter <= 0:
            counter = 15 
            
            if current_move == ACTION_FIREUP: current_move = ACTION_FIRERIGHT
            elif current_move == ACTION_FIRERIGHT: current_move = ACTION_FIREDOWN
            elif current_move == ACTION_FIREDOWN: current_move = ACTION_FIRELEFT
            elif current_move == ACTION_FIRELEFT: current_move = ACTION_FIREUP
            else: current_move = ACTION_FIREUP 
        
        bot_state[4] = current_move
        bot_state[5] = counter
        
        
        return current_move, bot_state
    
    def _get_wall_hugger_move(self, bot_state, shot_ok):
        """Implements a perimeter-hugging bot"""
        current_move = int(bot_state[4]) 
        counter = bot_state[5]
        pos_x, pos_y = bot_state[1:3]
        
        action_to_return = current_move 
        
        
        if current_move == ACTION_RIGHT and pos_x >= self.max_x:
            current_move = ACTION_DOWN
        elif current_move == ACTION_DOWN and pos_y >= self.max_y:
            current_move = ACTION_LEFT
        elif current_move == ACTION_LEFT and pos_x <= self.min_x:
            current_move = ACTION_UP
        elif current_move == ACTION_UP and pos_y <= self.min_y:
            current_move = ACTION_RIGHT
        
        action_to_return = current_move
        
        
        counter -= 1
        if counter <= 0 and shot_ok:
            counter = 20 
            if current_move == ACTION_RIGHT: action_to_return = ACTION_FIREDOWN 
            elif current_move == ACTION_DOWN: action_to_return = ACTION_FIRELEFT 
            elif current_move == ACTION_LEFT: action_to_return = ACTION_FIREUP 
            elif current_move == ACTION_UP: action_to_return = ACTION_FIRERIGHT 
            
        bot_state[4] = current_move 
        bot_state[5] = counter
        
        return action_to_return, bot_state

    def step(self, action):
        self.current_step += 1
        reward = self.survival_bonus
        terminated = False
        truncated = False
        
        
        old_agent_pos = self.agent_pos.copy()
        
        
        if action == ACTION_UP: self.agent_pos[1] -= self.bot_speed
        elif action == ACTION_DOWN: self.agent_pos[1] += self.bot_speed
        elif action == ACTION_LEFT: self.agent_pos[0] -= self.bot_speed
        elif action == ACTION_RIGHT: self.agent_pos[0] += self.bot_speed
        
        elif action >= ACTION_FIREUP and action <= ACTION_FIRERIGHT:
            if self.agent_ammo > 0:
                spawn_x, spawn_y = self.agent_pos[0] + self.bot_radius, self.agent_pos[1] + self.bot_radius
                dx, dy = 0, 0
                if action == ACTION_FIREUP: 
                    spawn_y = self.agent_pos[1] - 1
                    dy = -self.bullet_speed
                elif action == ACTION_FIREDOWN: 
                    spawn_y = self.agent_pos[1] + self.bot_diameter + 1
                    dy = self.bullet_speed
                elif action == ACTION_FIRELEFT: 
                    spawn_x = self.agent_pos[0] - 1
                    dx = -self.bullet_speed
                elif action == ACTION_FIRERIGHT: 
                    spawn_x = self.agent_pos[0] + self.bot_diameter + 1
                    dx = self.bullet_speed
                
                if self._add_bullet(spawn_x, spawn_y, dx, dy, 0): 
                    self.agent_ammo -= 1
        
        

        
        
        hit_wall = False
        if not (self.min_x <= self.agent_pos[0] <= self.max_x):
            hit_wall = True
        if not (self.min_y <= self.agent_pos[1] <= self.max_y):
            hit_wall = True
        
        if hit_wall:
            self.agent_pos = np.clip(self.agent_pos, [self.min_x, self.min_y], [self.max_x, self.max_y])
            reward += self.wall_penalty
            
        
        if self.num_bots_alive > 0:
            agent_center = self.agent_pos + self.bot_radius
            bot_centers = self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
            distances = np.linalg.norm(bot_centers - agent_center, axis=1)
            
            if np.any(distances < self.bot_diameter):
                self.agent_pos = old_agent_pos 
                reward += self.wall_penalty 

        
        bots_to_remove = []
        for i in range(self.num_bots_alive):
            
            bot_state = self.other_bots_state[i]
            bot_type = int(bot_state[0])
            old_bot_pos = bot_state[1:3].copy()
            shot_ok = bot_state[3] > 0

            
            if bot_type == self.BOT_TYPE_RAND:
                action, bot_state = self._get_rand_move(bot_state)
            elif bot_type == self.BOT_TYPE_HUNTER:
                action, bot_state = self._get_hunter_move(i, bot_state, shot_ok)
            elif bot_type == self.BOT_TYPE_SPINNER:
                action, bot_state = self._get_spinner_move(bot_state)
            elif bot_type == self.BOT_TYPE_WALL_HUGGER:
                action, bot_state = self._get_wall_hugger_move(bot_state, shot_ok)
            
            
            self.other_bots_state[i] = bot_state
            
            
            bot_pos = bot_state[1:3] 
            
            if action == ACTION_UP: bot_pos[1] -= self.bot_speed
            elif action == ACTION_DOWN: bot_pos[1] += self.bot_speed
            elif action == ACTION_LEFT: bot_pos[0] -= self.bot_speed
            elif action == ACTION_RIGHT: bot_pos[0] += self.bot_speed
            elif action >= ACTION_FIREUP and action <= ACTION_FIRERIGHT:
                if shot_ok: 
                    spawn_x, spawn_y = bot_pos[0] + self.bot_radius, bot_pos[1] + self.bot_radius
                    dx, dy = 0, 0
                    if action == ACTION_FIREUP:
                        spawn_y = bot_pos[1] - 1; dy = -self.bullet_speed
                    elif action == ACTION_FIREDOWN:
                        spawn_y = bot_pos[1] + self.bot_diameter + 1; dy = self.bullet_speed
                    elif action == ACTION_FIRELEFT:
                        spawn_x = bot_pos[0] - 1; dx = -self.bullet_speed
                    elif action == ACTION_FIRERIGHT:
                        spawn_x = bot_pos[0] + self.bot_diameter + 1; dx = self.bullet_speed

                    if self._add_bullet(spawn_x, spawn_y, dx, dy, i + 1): 
                        self.other_bots_state[i, 3] -= 1 
            
            
            
            bot_pos = np.clip(bot_pos, [self.min_x, self.min_y], [self.max_x, self.max_y])
            
            
            all_centers = np.vstack([
                self.agent_pos + self.bot_radius,
                self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
            ])
            my_center = bot_pos + self.bot_radius
            my_idx = i + 1 
            
            distances = np.linalg.norm(all_centers - my_center, axis=1)
            distances[my_idx] = np.inf 
            
            if np.any(distances < self.bot_diameter):
                bot_pos = old_bot_pos 
            
            
            self.other_bots_state[i, 1:3] = bot_pos

        
        bullets_to_keep = np.ones(self.num_bullets_active, dtype=bool)
        ammo_to_refund = {} 

        agent_center = self.agent_pos + self.bot_radius
        if self.num_bots_alive > 0:
            bot_centers = self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
        
        for i in range(self.num_bullets_active):
            bullet = self.bullets[i]
            bullet[:2] += bullet[2:4] 
            owner_id = int(bullet[4])
            
            
            if not (self.min_x <= bullet[0] <= self.width and self.min_y <= bullet[1] <= self.height):
                bullets_to_keep[i] = False
                ammo_to_refund[owner_id] = ammo_to_refund.get(owner_id, 0) + 1
                continue
                
            
            
            hit = False
            if owner_id == 0: 
                if self.num_bots_alive > 0:
                    distances = np.linalg.norm(bot_centers - bullet[:2], axis=1)
                    hit_indices = np.where(distances < self.bot_radius)[0]
                    if len(hit_indices) > 0:
                        hit_bot_idx = hit_indices[0]
                        bots_to_remove.append(hit_bot_idx)
                        reward += self.kill_reward
                        hit = True
            else: 
                distance_to_agent = np.linalg.norm(agent_center - bullet[:2])
                if distance_to_agent < self.bot_radius:
                    terminated = True
                    reward = self.death_penalty
                    hit = True
            
            if hit:
                bullets_to_keep[i] = False
                ammo_to_refund[owner_id] = ammo_to_refund.get(owner_id, 0) + 1

        
        if bots_to_remove:
            
            unique_indices = sorted(list(set(bots_to_remove)), reverse=True)
            for idx in unique_indices:
                
                self.other_bots_state[idx] = self.other_bots_state[self.num_bots_alive - 1]
                
                self.other_bots_state[self.num_bots_alive - 1] = 0 
                self.num_bots_alive -= 1

        
        
        kept_bullets_array = self.bullets[:self.num_bullets_active][bullets_to_keep]
        
        
        self.num_bullets_active = np.sum(bullets_to_keep)
        
        
        self.bullets[:self.num_bullets_active] = kept_bullets_array
        
        self.bullets[self.num_bullets_active:] = 0
        
        for owner_id, count in ammo_to_refund.items():
            if owner_id == 0:
                self.agent_ammo = min(self.agent_ammo + count, self.max_bullets_per_bot)
            else: 
                bot_idx = owner_id - 1
                
                if bot_idx < self.num_bots_alive:
                     self.other_bots_state[bot_idx, 3] = min(
                         self.other_bots_state[bot_idx, 3] + count, 
                         self.max_bullets_per_bot
                     )

        
        if self.current_step >= self.max_steps:
            truncated = True
            
        if self.num_bots_alive == 0 and not terminated:
            terminated = True
            reward = self.win_reward
            
        return self._get_obs(), reward, terminated, truncated, {}


def initialize_gui():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BattleBots AI")
    return screen, pygame.time.Clock()

def draw_entities(screen, env):
    
    
    
    pygame.draw.rect(screen, COLOR_WALL, (ARENA_LEFT, ARENA_TOP, ARENA_WIDTH, ARENA_HEIGHT - ARENA_TOP), 1)

    
    for i in range(env.num_bots_alive):
        bot_pos = env.other_bots_state[i, 1:3] 
        center = (int(bot_pos[0] + env.bot_radius), int(bot_pos[1] + env.bot_radius))
        pygame.draw.circle(screen, COLOR_BOT, center, int(env.bot_radius))

    
    for i in range(env.num_bullets_active):
        bullet = env.bullets[i]
        pos = (int(bullet[0]), int(bullet[1]))
        color = COLOR_BULLET_AGENT if int(bullet[4]) == 0 else COLOR_BULLET_BOT
        
        if bullet[2] == 0: 
            pygame.draw.line(screen, color, (pos[0], pos[1] - 3), (pos[0], pos[1] + 3), 3)
        else: 
            pygame.draw.line(screen, color, (pos[0] - 3, pos[1]), (pos[0] + 3, pos[1]), 3)

    
    agent_center = (int(env.agent_pos[0] + env.bot_radius), int(env.agent_pos[1] + env.bot_radius))
    pygame.draw.circle(screen, COLOR_AGENT, agent_center, int(env.bot_radius))
    
    pygame.draw.circle(screen, (200, 220, 255), agent_center, int(env.bot_radius), 3)

def run_gui(model, env_kwargs, seed=None):
    env = BattleBotsEnv(**env_kwargs)
    screen, clock = initialize_gui()
    
    
    
    
    obs, _ = env.reset(seed=seed)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
        
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        screen.fill(COLOR_BG)
        draw_entities(screen, env)
        pygame.display.flip()
        
        if terminated or truncated:
            print("Episode Finished. Resetting...")
            if terminated and env.num_bots_alive == 0:
                print("--- AGENT WON ---")
            elif terminated:
                print("--- AGENT KILLED ---")
            time.sleep(1.0)
            obs, _ = env.reset(seed=seed) 
        clock.tick(FRAME_RATE)
    pygame.quit()

if __name__ == '__main__':
    
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or visualize a PPO agent for BattleBots.")
    
    
    
    
    
    
    
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (for the linear schedule).')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor gamma.')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='Factor for trade-off of bias vs. variance for GAE.')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='Clipping parameter, as a float.')
    parser.add_argument('--ent_coef', type=float, default=0.005,
                        help='Entropy coefficient for the loss calculation.')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function coefficient for the loss calculation.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='The maximum value for the gradient clipping.')
    
    
    
    
    parser.add_argument('--model_name', type=str, default=MODEL_FILENAME,
                        help='Filename for saving the trained model.')
    parser.add_argument('--log_dir', type=str, default="./ppo_battlebots_tensorboard/",
                        help='Tensorboard log directory.')
    
    parser.add_argument('--train_model', type=str_to_bool, default=True,
                        help='Enables Visualization / Training mode (e.g., true or false)')
    
    
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible training. If not set, run will be non-deterministic.')

    args = parser.parse_args()

    if args.train_model:
        num_cpu = max(1, cpu_count() - 1) * 2 
        print(f"--- Starting Training using {num_cpu} cores ---")

        
        
        
        vec_env_kwargs = {"env_kwargs": TRAINING_KWARGS}
        
        if args.seed is not None:
            print(f"--- Setting random seed to: {args.seed} ---")
            set_random_seed(args.seed)
            vec_env_kwargs['seed'] = args.seed
        else:
            print("--- No seed set, running non-deterministically ---")
            
            
            
            
        
        
        
        run_log_dir = os.path.join(args.log_dir, os.path.splitext(args.model_name)[0])
        print(f"Logging to: {run_log_dir}")
        print(f"Saving model to: {args.model_name}")
        
        
        env = make_vec_env(BattleBotsEnv, n_envs=num_cpu,
                           vec_env_cls=SubprocVecEnv, 
                           **vec_env_kwargs) 
        
        
        print("Creating evaluation environment...")
        
        eval_kwargs = {"env_kwargs": TESTING_KWARGS}
        if args.seed is not None:
            
            eval_kwargs['seed'] = args.seed + 1 
        
        
        eval_env = make_vec_env(BattleBotsEnv, n_envs=1, **eval_kwargs)
        
        
        policy_kwargs = dict(net_arch=dict(pi=[256, 512, 256], vf=[256, 256]))
        
        model = PPO("MlpPolicy", env, verbose=1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=run_log_dir,  
                    
                    
                    
                    
                    learning_rate=linear_schedule(args.lr),
                    
                    
                    n_steps=2048,
                    n_epochs=10,
                    batch_size=64,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    
                    
                    )
        
        
        
        best_model_save_dir = os.path.join(run_log_dir, "best_model")
        print(f"Best model will be saved to: {best_model_save_dir}")

        eval_callback = EvalCallback(eval_env, 
                                 best_model_save_path=best_model_save_dir,
                                 log_path=best_model_save_dir, 
                                 eval_freq=10000, 
                                 n_eval_episodes=5, 
                                 deterministic=True,
                                 render=False)
        
        
        print(f"\n--- RUNNING WITH PARAMS ---")
        print(f"  lr: {args.lr}")
        print(f"  gamma: {args.gamma}")
        print(f"  gae_lambda: {args.gae_lambda}")
        print(f"  clip_range: {args.clip_range}")
        print(f"  ent_coef: {args.ent_coef}")
        print(f"  vf_coef: {args.vf_coef}")
        print(f"  max_grad_norm: {args.max_grad_norm}")
        print(f"  seed: {args.seed}") 
        print("-----------------------------\n")
        
        
        model.learn(total_timesteps=1_000_000, progress_bar=True, callback=eval_callback)
        model.save(args.model_name) 
        
        print(f"--- Training Finished. Final model saved to {args.model_name} ---")
        print(f"Best model saved in {best_model_save_dir}")
        print("Set TRAIN_MODEL to False to watch the agent.")
    else:
        
        
        watch_model = MODEL_FILENAME
        if args.model_name != MODEL_FILENAME:
            watch_model = args.model_name
            print(f"[Info] --model_name specified, overriding default.")
        
        print(f"--- Starting GUI Visualization (loading {watch_model}) ---")
        if not os.path.exists(watch_model):
            print(f"Error: Model file '{watch_model}' not found. Please train first.")
        else:
            model = PPO.load(watch_model)
            run_gui(model, env_kwargs=TESTING_KWARGS, seed=args.seed)
# main.py

import os
import time
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from multiprocessing import cpu_count
from typing import Callable, List, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# --- SCRIPT CONTROLS ---
TRAIN_MODEL = True # Set to False to watch a trained model with the GUI
MODEL_FILENAME = "ppo_battlebots_agent_v2.zip"

# --- JAVA ENVIRONMENT CONSTANTS ---
# These are drawn from BattleBotArena.java
ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 700.0
ARENA_TOP = 10.0
ARENA_LEFT = 0.0
BOT_RADIUS = 13.0
BOT_DIAMETER = BOT_RADIUS * 2.0
BOT_SPEED = 3.0
BULLET_SPEED = 3.0
MAX_BULLETS_PER_BOT = 4
TIME_LIMIT_SECS = 110
FRAMES_PER_SEC = 30
MAX_STEPS = TIME_LIMIT_SECS * FRAMES_PER_SEC # 90 seconds * 30fps

# --- ACTIONS (from Java constants) ---
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_FIREUP = 4
ACTION_FIREDOWN = 5
ACTION_FIRELEFT = 6
ACTION_FIRERIGHT = 7
ACTION_STAY = 8

# --- CURRICULUM & ENVIRONMENT SETTINGS ---
# Use these simpler settings for TRAINING the model
TRAINING_KWARGS = {
    'width': ARENA_WIDTH,
    'height': ARENA_HEIGHT,
    'max_steps': 1000, # Shorter episodes for faster training
    'min_bots': 8,
    'max_bots': 12,
    'max_bullets_per_bot': MAX_BULLETS_PER_BOT,
    'bot_radius': BOT_RADIUS,
    'bot_speed': BOT_SPEED,
    'bullet_speed': BULLET_SPEED,
    'win_reward': 15.0,            # Bonus for killing all other bots
    'death_penalty': -12.0,        # Penalty for being killed
    'kill_reward': 10.0,            # From KILL_SCORE
    'survival_bonus_per_step': 0.3 / FRAMES_PER_SEC, # From POINTS_PER_SECOND
    'wall_penalty': -0.2           # Penalty for hitting wall or bot
}
# Use these harder settings for WATCHING the trained model
TESTING_KWARGS = {
    'width': ARENA_WIDTH,
    'height': ARENA_HEIGHT,
    'max_steps': MAX_STEPS,
    'min_bots': 8,
    'max_bots': 12,
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
# --- GUI CONSTANTS ---
# Screen size matches arena size
SCREEN_WIDTH = int(ARENA_WIDTH)
SCREEN_HEIGHT = int(ARENA_HEIGHT)
FRAME_RATE = 30 # Run GUI at the same speed as the game logic
# Colors
COLOR_BG = (10, 10, 10); COLOR_WALL = (40, 40, 40); COLOR_AGENT = (50, 150, 255)
COLOR_BOT = (255, 150, 50); COLOR_BULLET_AGENT = (100, 200, 255); COLOR_BULLET_BOT = (255, 150, 150)
COLOR_DEAD = (100, 100, 100)

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

class BattleBotsEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, width=1000.0, height=700.0, max_steps=2700, min_bots=1,
                 max_bots=5, max_bullets_per_bot=4, bot_radius=13.0,
                 bot_speed=3.0, bullet_speed=6.0, win_reward=10.0,
                 death_penalty=-10.0, kill_reward=5.0,
                 survival_bonus_per_step=0.0033, wall_penalty=-0.2):
        super().__init__()
        # --- Store all parameters ---
        self.width = float(width); self.height = float(height)
        self.max_steps = int(max_steps); self.min_bots = int(min_bots)
        self.max_bots = int(max_bots); self.max_bullets_per_bot = int(max_bullets_per_bot)
        self.bot_radius = float(bot_radius); self.bot_diameter = self.bot_radius * 2.0
        self.bot_speed = float(bot_speed); self.bullet_speed = float(bullet_speed)
        
        # --- Rewards ---
        self.win_reward = float(win_reward); self.death_penalty = float(death_penalty)
        self.kill_reward = float(kill_reward)
        self.survival_bonus = float(survival_bonus_per_step)
        self.wall_penalty = float(wall_penalty)
        
        # --- Arena Boundaries ---
        self.min_x = ARENA_LEFT
        self.max_x = self.width - self.bot_diameter
        self.min_y = ARENA_TOP
        self.max_y = self.height - self.bot_diameter

        # --- Bot Types ---
        self.BOT_TYPE_RAND = 0
        self.BOT_TYPE_HUNTER = 1
        self.BOT_TYPE_SPINNER = 2
        self.BOT_TYPE_WALL_HUGGER = 3

        # --- Define Spaces ---
        # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT, 4:FIREUP, 5:FIREDOWN, 6:FIRELEFT, 7:FIRERIGHT, 8:STAY
        self.action_space = spaces.Discrete(9)
        
        self.max_total_bullets = (self.max_bots + 1) * self.max_bullets_per_bot
        
        # Observation:
        # 1. Agent pos (x, y) - normalized (2)
        # 2. Agent ammo (n) - normalized (1)
        # 3. Other bots' info (x, y, ammo) - normalized (max_bots * 3)
        # 4. All bullets' pos (x, y) - normalized (max_total_bullets * 2)
        # We stack 2 frames, as in the original script
        obs_frame_len = 2 + 1 + (self.max_bots * 3) + (self.max_total_bullets * 2)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(obs_frame_len * 2,), dtype=np.float32
        )
        
        # --- Internal State ---
        self.prev_obs = None
        self.current_step = 0
        self.agent_pos = np.zeros(2, dtype=np.float32)
        self.agent_ammo = 0
        
        # Other bots state array:
        # 0: bot_type (0=Rand, 1=Hunter, 2=Spinner, 3=WallHugger)
        # 1: x
        # 2: y
        # 3: ammo
        # 4: current_move (action 0-8)
        # 5: counter (multi-purpose counter)
        # 6: target_x (for Hunter)
        # 7: target_y (for Hunter)
        self.other_bots_state = np.zeros((self.max_bots, 8), dtype=np.float32)
        self.num_bots_alive = 0
        
        # Bullets: [x, y, dx, dy, owner_id (0=agent, 1-N=bot)]
        self.bullets = np.zeros((self.max_total_bullets, 5), dtype=np.float32)
        self.num_bullets_active = 0
        
    def _get_obs(self):
        # 1. Agent pos (2)
        agent_pos_norm = np.array([
            (self.agent_pos[0] - self.min_x) / (self.max_x - self.min_x) * 2 - 1,
            (self.agent_pos[1] - self.min_y) / (self.max_y - self.min_y) * 2 - 1
        ], dtype=np.float32)
        
        # 2. Agent ammo (1)
        agent_ammo_norm = np.array([(self.agent_ammo / self.max_bullets_per_bot) * 2 - 1], dtype=np.float32)
        
        # 3. Other bots info (max_bots * 3)
        bot_info_padded = np.full((self.max_bots, 3), -1.0, dtype=np.float32)
        if self.num_bots_alive > 0:
            live_bots = self.other_bots_state[:self.num_bots_alive]
            # Read from new state array: pos=[1, 2], ammo=[3]
            bot_pos_norm = np.array([
                (live_bots[:, 1] - self.min_x) / (self.max_x - self.min_x) * 2 - 1,
                (live_bots[:, 2] - self.min_y) / (self.max_y - self.min_y) * 2 - 1
            ]).T
            bot_ammo_norm = (live_bots[:, 3] / self.max_bullets_per_bot) * 2 - 1
            bot_info_padded[:self.num_bots_alive, :2] = bot_pos_norm
            bot_info_padded[:self.num_bots_alive, 2] = bot_ammo_norm
        
        # 4. Bullets info (max_total_bullets * 2)
        bullet_info_padded = np.full((self.max_total_bullets, 2), -1.0, dtype=np.float32)
        if self.num_bullets_active > 0:
            active_bullets = self.bullets[:self.num_bullets_active, :2]
            bullet_pos_norm = np.array([
                (active_bullets[:, 0] - self.min_x) / (self.width - self.min_x) * 2 - 1,
                (active_bullets[:, 1] - self.min_y) / (self.height - self.min_y) * 2 - 1
            ]).T
            bullet_info_padded[:self.num_bullets_active] = bullet_pos_norm

        # Concatenate all
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
        
        # Spawn agent and bots without overlap (Java reset logic)
        positions = []
        spawn_count = self.num_bots_alive + 1 # bots + agent
        x_scale = (self.width - self.bot_diameter * 2) / max(spawn_count - 1, 1)
        y_scale = (self.height - self.bot_diameter * 2) / 5 # 5 rows
        
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
            bot_type = self.np_random.integers(0, 4) # 0=Rand, 1=Hunter, 2=Spinner, 3=WallHugger
            
            self.other_bots_state[i, 0] = bot_type
            self.other_bots_state[i, 1] = pos[0] # x
            self.other_bots_state[i, 2] = pos[1] # y
            self.other_bots_state[i, 3] = self.max_bullets_per_bot # ammo
            
            if bot_type == self.BOT_TYPE_RAND:
                self.other_bots_state[i, 4] = self.np_random.integers(0, 4) # current_move
                self.other_bots_state[i, 5] = 99 # counter (moveCount)
            elif bot_type == self.BOT_TYPE_HUNTER:
                self.other_bots_state[i, 4] = ACTION_STAY
                self.other_bots_state[i, 5] = 0 # Target re-acquisition counter
            elif bot_type == self.BOT_TYPE_SPINNER:
                self.other_bots_state[i, 4] = ACTION_FIREUP
                self.other_bots_state[i, 5] = 15 # Fire delay counter
            elif bot_type == self.BOT_TYPE_WALL_HUGGER:
                self.other_bots_state[i, 4] = ACTION_RIGHT # Start moving right
                self.other_bots_state[i, 5] = 20 # Fire counter
            
        return self._get_obs(), {}

    def _get_rand_move(self, bot_state):
        """Implements RandBot logic"""
        current_move = int(bot_state[4])
        move_count = bot_state[5]
        
        move_count += 1
        
        # Time to choose a new move?
        if move_count >= 30 + self.np_random.integers(0, 60):
            move_count = 0
            choice = self.np_random.integers(0, 8) # 0-7
            
            if choice == 0: current_move = ACTION_UP
            elif choice == 1: current_move = ACTION_DOWN
            elif choice == 2: current_move = ACTION_LEFT
            elif choice == 3: current_move = ACTION_RIGHT
            elif choice == 4: current_move = ACTION_FIREUP
            elif choice == 5: current_move = ACTION_FIREDOWN
            elif choice == 6: current_move = ACTION_FIRELEFT
            elif choice == 7: current_move = ACTION_FIRERIGHT
            
            # make sure we choose a new move next time after firing
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
        
        # Re-acquire target every 20 steps
        counter += 1
        if counter > 20 or np.array_equal(target_pos, [0, 0]):
            counter = 0
            
            # Build list of potential targets
            targets = []
            # Agent is target 0
            targets.append((self.agent_pos + self.bot_radius, 0))
            # Other bots are targets 1+
            for i in range(self.num_bots_alive):
                if i != bot_idx:
                    targets.append((self.other_bots_state[i, 1:3] + self.bot_radius, i + 1))
            
            # Find closest target
            min_dist = float('inf')
            best_target = None
            for t_pos, t_id in targets:
                dist = np.linalg.norm(bot_center - t_pos)
                if dist < min_dist:
                    min_dist = dist
                    best_target = t_pos
            
            if best_target is not None:
                target_pos = best_target # Store center pos of target
            bot_state[6:8] = target_pos

        bot_state[5] = counter
        
        # --- Decide move based on target ---
        dx = target_pos[0] - bot_center[0]
        dy = target_pos[1] - bot_center[1]
        
        alignment_threshold = self.bot_radius * 1.5
        
        # 1. Fire if aligned
        if shot_ok:
            if abs(dy) < alignment_threshold: # Horizontally aligned
                if dx > 0: return ACTION_FIRERIGHT, bot_state
                else: return ACTION_FIRELEFT, bot_state
            if abs(dx) < alignment_threshold: # Vertically aligned
                if dy > 0: return ACTION_FIREDOWN, bot_state
                else: return ACTION_FIREUP, bot_state
                
        # 2. Move to align
        if abs(dx) > abs(dy): # Move horizontally
            if dx > 0: return ACTION_RIGHT, bot_state
            else: return ACTION_LEFT, bot_state
        else: # Move vertically
            if dy > 0: return ACTION_DOWN, bot_state
            else: return ACTION_UP, bot_state

    def _get_spinner_move(self, bot_state):
        """Implements a stationary spinner bot"""
        current_move = int(bot_state[4])
        counter = bot_state[5]
        
        counter -= 1
        
        if counter <= 0:
            counter = 15 # Reset fire delay
            # Rotate firing direction
            if current_move == ACTION_FIREUP: current_move = ACTION_FIRERIGHT
            elif current_move == ACTION_FIRERIGHT: current_move = ACTION_FIREDOWN
            elif current_move == ACTION_FIREDOWN: current_move = ACTION_FIRELEFT
            elif current_move == ACTION_FIRELEFT: current_move = ACTION_FIREUP
            else: current_move = ACTION_FIREUP # Default
        
        bot_state[4] = current_move
        bot_state[5] = counter
        
        # Spinner only fires, it doesn't move
        return current_move, bot_state
    
    def _get_wall_hugger_move(self, bot_state, shot_ok):
        """Implements a perimeter-hugging bot"""
        current_move = int(bot_state[4]) # This is the *movement* direction
        counter = bot_state[5]
        pos_x, pos_y = bot_state[1:3]
        
        action_to_return = current_move # Default: keep moving
        
        # 1. Check for wall collisions to change direction
        if current_move == ACTION_RIGHT and pos_x >= self.max_x:
            current_move = ACTION_DOWN
        elif current_move == ACTION_DOWN and pos_y >= self.max_y:
            current_move = ACTION_LEFT
        elif current_move == ACTION_LEFT and pos_x <= self.min_x:
            current_move = ACTION_UP
        elif current_move == ACTION_UP and pos_y <= self.min_y:
            current_move = ACTION_RIGHT
        
        action_to_return = current_move
        
        # 2. Fire inwards every N steps
        counter -= 1
        if counter <= 0 and shot_ok:
            counter = 20 # Reset fire delay
            if current_move == ACTION_RIGHT: action_to_return = ACTION_FIREDOWN # Fire down (inwards)
            elif current_move == ACTION_DOWN: action_to_return = ACTION_FIRELEFT # Fire left (inwards)
            elif current_move == ACTION_LEFT: action_to_return = ACTION_FIREUP # Fire up (inwards)
            elif current_move == ACTION_UP: action_to_return = ACTION_FIRERIGHT # Fire right (inwards)
            
        bot_state[4] = current_move # Store the *movement* direction
        bot_state[5] = counter
        
        return action_to_return, bot_state

    def step(self, action):
        self.current_step += 1
        reward = self.survival_bonus
        terminated = False
        truncated = False
        
        # --- 1. Agent Action ---
        old_agent_pos = self.agent_pos.copy()
        
        # 0:UP, 1:DOWN, 2:LEFT, 3:RIGHT
        if action == ACTION_UP: self.agent_pos[1] -= self.bot_speed
        elif action == ACTION_DOWN: self.agent_pos[1] += self.bot_speed
        elif action == ACTION_LEFT: self.agent_pos[0] -= self.bot_speed
        elif action == ACTION_RIGHT: self.agent_pos[0] += self.bot_speed
        # 4:FIREUP, 5:FIREDOWN, 6:FIRELEFT, 7:FIRERIGHT
        elif action >= ACTION_FIREUP and action <= ACTION_FIRERIGHT:
            if self.agent_ammo > 0:
                spawn_x, spawn_y = self.agent_pos[0] + self.bot_radius, self.agent_pos[1] + self.bot_radius
                dx, dy = 0, 0
                if action == ACTION_FIREUP: # UP
                    spawn_y = self.agent_pos[1] - 1
                    dy = -self.bullet_speed
                elif action == ACTION_FIREDOWN: # DOWN
                    spawn_y = self.agent_pos[1] + self.bot_diameter + 1
                    dy = self.bullet_speed
                elif action == ACTION_FIRELEFT: # LEFT
                    spawn_x = self.agent_pos[0] - 1
                    dx = -self.bullet_speed
                elif action == ACTION_FIRERIGHT: # RIGHT
                    spawn_x = self.agent_pos[0] + self.bot_diameter + 1
                    dx = self.bullet_speed
                
                if self._add_bullet(spawn_x, spawn_y, dx, dy, 0): # 0 = agent
                    self.agent_ammo -= 1
        # 8: STAY
        # else: pass

        # --- 2. Agent Collision Check (Walls & Bots) ---
        # 2a. Walls
        hit_wall = False
        if not (self.min_x <= self.agent_pos[0] <= self.max_x):
            hit_wall = True
        if not (self.min_y <= self.agent_pos[1] <= self.max_y):
            hit_wall = True
        
        if hit_wall:
            self.agent_pos = np.clip(self.agent_pos, [self.min_x, self.min_y], [self.max_x, self.max_y])
            reward += self.wall_penalty
            
        # 2b. Other Bots
        if self.num_bots_alive > 0:
            agent_center = self.agent_pos + self.bot_radius
            bot_centers = self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
            distances = np.linalg.norm(bot_centers - agent_center, axis=1)
            
            if np.any(distances < self.bot_diameter):
                self.agent_pos = old_agent_pos # Revert move
                reward += self.wall_penalty # Penalty for bot collision

        # --- 3. Other Bots AI & Movement ---
        bots_to_remove = []
        for i in range(self.num_bots_alive):
            # Get current state
            bot_state = self.other_bots_state[i]
            bot_type = int(bot_state[0])
            old_bot_pos = bot_state[1:3].copy()
            shot_ok = bot_state[3] > 0

            # Get action from the correct AI
            if bot_type == self.BOT_TYPE_RAND:
                action, bot_state = self._get_rand_move(bot_state)
            elif bot_type == self.BOT_TYPE_HUNTER:
                action, bot_state = self._get_hunter_move(i, bot_state, shot_ok)
            elif bot_type == self.BOT_TYPE_SPINNER:
                action, bot_state = self._get_spinner_move(bot_state)
            elif bot_type == self.BOT_TYPE_WALL_HUGGER:
                action, bot_state = self._get_wall_hugger_move(bot_state, shot_ok)
            
            # Store the updated state (e.g., new counters, new move)
            self.other_bots_state[i] = bot_state
            
            # --- Process the bot's chosen action ---
            bot_pos = bot_state[1:3] # Get pos from state
            
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

                    if self._add_bullet(spawn_x, spawn_y, dx, dy, i + 1): # owner_id = 1+
                        self.other_bots_state[i, 3] -= 1 # Decrement ammo in state
            
            # --- Bot Collision Check ---
            # Walls
            bot_pos = np.clip(bot_pos, [self.min_x, self.min_y], [self.max_x, self.max_y])
            
            # Other bots and agent
            all_centers = np.vstack([
                self.agent_pos + self.bot_radius,
                self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
            ])
            my_center = bot_pos + self.bot_radius
            my_idx = i + 1 # 0 is agent, 1+ are bots
            
            distances = np.linalg.norm(all_centers - my_center, axis=1)
            distances[my_idx] = np.inf # Don't check against self
            
            if np.any(distances < self.bot_diameter):
                bot_pos = old_bot_pos # Revert move
            
            # Save final position to state
            self.other_bots_state[i, 1:3] = bot_pos

        # --- 4. Process Bullets (Movement & Collisions) ---
        bullets_to_keep = np.ones(self.num_bullets_active, dtype=bool)
        ammo_to_refund = {} # {owner_id: count}

        agent_center = self.agent_pos + self.bot_radius
        if self.num_bots_alive > 0:
            bot_centers = self.other_bots_state[:self.num_bots_alive, 1:3] + self.bot_radius
        
        for i in range(self.num_bullets_active):
            bullet = self.bullets[i]
            bullet[:2] += bullet[2:4] # Move bullet
            owner_id = int(bullet[4])
            
            # 4a. Off-screen check
            if not (self.min_x <= bullet[0] <= self.width and self.min_y <= bullet[1] <= self.height):
                bullets_to_keep[i] = False
                ammo_to_refund[owner_id] = ammo_to_refund.get(owner_id, 0) + 1
                continue
                
            # 4b. Bullet-Bot/Agent Collision Check
            # Check distance from bullet (point) to bot/agent center
            hit = False
            if owner_id == 0: # Agent's bullet
                if self.num_bots_alive > 0:
                    distances = np.linalg.norm(bot_centers - bullet[:2], axis=1)
                    hit_indices = np.where(distances < self.bot_radius)[0]
                    if len(hit_indices) > 0:
                        hit_bot_idx = hit_indices[0]
                        bots_to_remove.append(hit_bot_idx)
                        reward += self.kill_reward
                        hit = True
            else: # A bot's bullet
                distance_to_agent = np.linalg.norm(agent_center - bullet[:2])
                if distance_to_agent < self.bot_radius:
                    terminated = True
                    reward = self.death_penalty
                    hit = True
            
            if hit:
                bullets_to_keep[i] = False
                ammo_to_refund[owner_id] = ammo_to_refund.get(owner_id, 0) + 1

        # --- 5. Clean up dead bots ---
        if bots_to_remove:
            # Remove duplicates and sort descending to avoid index errors
            unique_indices = sorted(list(set(bots_to_remove)), reverse=True)
            for idx in unique_indices:
                # Swap with last bot and decrement count
                self.other_bots_state[idx] = self.other_bots_state[self.num_bots_alive - 1]
                # Clear the (now duplicate) last bot's state
                self.other_bots_state[self.num_bots_alive - 1] = 0 
                self.num_bots_alive -= 1

        # --- 6. Clean up bullets and refund ammo ---
        # First, get the array of bullets we are keeping
        kept_bullets_array = self.bullets[:self.num_bullets_active][bullets_to_keep]
        
        # Now, update the active bullet count to the new, smaller number
        self.num_bullets_active = np.sum(bullets_to_keep)
        
        # Finally, assign the kept bullets to the *correctly sized slice* at the beginning
        self.bullets[:self.num_bullets_active] = kept_bullets_array
        # Clear the old (now duplicate) bullet data
        self.bullets[self.num_bullets_active:] = 0
        
        for owner_id, count in ammo_to_refund.items():
            if owner_id == 0:
                self.agent_ammo = min(self.agent_ammo + count, self.max_bullets_per_bot)
            else: # Bot
                bot_idx = owner_id - 1
                # Check if bot is still alive (index might be invalid if bot was killed)
                if bot_idx < self.num_bots_alive:
                     self.other_bots_state[bot_idx, 3] = min(
                         self.other_bots_state[bot_idx, 3] + count, 
                         self.max_bullets_per_bot
                     )

        # --- 7. Check final terminal conditions ---
        if self.current_step >= self.max_steps:
            truncated = True
            
        if self.num_bots_alive == 0 and not terminated:
            terminated = True
            reward = self.win_reward
            
        return self._get_obs(), reward, terminated, truncated, {}

# --- Pygame GUI Functions ---
def initialize_gui():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BattleBots AI")
    return screen, pygame.time.Clock()

def draw_entities(screen, env):
    # Pygame's (0,0) is top-left, same as the environment's. No Y-inversion needed.
    
    # Draw arena boundary
    pygame.draw.rect(screen, COLOR_WALL, (ARENA_LEFT, ARENA_TOP, ARENA_WIDTH, ARENA_HEIGHT - ARENA_TOP), 1)

    # Draw Bots
    for i in range(env.num_bots_alive):
        bot_pos = env.other_bots_state[i, 1:3] # Read x, y from state
        center = (int(bot_pos[0] + env.bot_radius), int(bot_pos[1] + env.bot_radius))
        pygame.draw.circle(screen, COLOR_BOT, center, int(env.bot_radius))

    # Draw Bullets
    for i in range(env.num_bullets_active):
        bullet = env.bullets[i]
        pos = (int(bullet[0]), int(bullet[1]))
        color = COLOR_BULLET_AGENT if int(bullet[4]) == 0 else COLOR_BULLET_BOT
        # Draw bullet as a thick line (like in Java)
        if bullet[2] == 0: # Vertical
            pygame.draw.line(screen, color, (pos[0], pos[1] - 3), (pos[0], pos[1] + 3), 3)
        else: # Horizontal
            pygame.draw.line(screen, color, (pos[0] - 3, pos[1]), (pos[0] + 3, pos[1]), 3)

    # Draw Agent
    agent_center = (int(env.agent_pos[0] + env.bot_radius), int(env.agent_pos[1] + env.bot_radius))
    pygame.draw.circle(screen, COLOR_AGENT, agent_center, int(env.bot_radius))
    # Draw outline (shows it's the agent)
    pygame.draw.circle(screen, (200, 220, 255), agent_center, int(env.bot_radius), 3)

def run_gui(model, env_kwargs):
    env = BattleBotsEnv(**env_kwargs)
    screen, clock = initialize_gui()
    obs, _ = env.reset()
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
            obs, _ = env.reset()
        clock.tick(FRAME_RATE)
    pygame.quit()

if __name__ == '__main__':
    
    # We add argparse to handle command-line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Train or visualize a PPO agent for BattleBots.")
    
    # --- Script Controls ---
    # We keep TRAIN_MODEL as a script constant, as it changes the
    # fundamental behavior (train vs. watch).
    
    # --- Hyperparameters for Tuning ---
    # We use the commented-out values from your script as the defaults.
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Initial learning rate (for the linear schedule).')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor gamma.')
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help='Factor for trade-off of bias vs. variance for GAE.')
    parser.add_argument('--clip_range', type=float, default=0.2,
                        help='Clipping parameter, as a float.')
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help='Entropy coefficient for the loss calculation.')
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help='Value function coefficient for the loss calculation.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='The maximum value for the gradient clipping.')
    
    # --- File/Log Management ---
    # These are CRITICAL for running multiple experiments,
    # so they don't overwrite each other.
    parser.add_argument('--model_name', type=str, default=MODEL_FILENAME,
                        help='Filename for saving the trained model.')
    parser.add_argument('--log_dir', type=str, default="./ppo_battlebots_tensorboard/",
                        help='Tensorboard log directory.')

    args = parser.parse_args()

    if TRAIN_MODEL:
        num_cpu = max(1, cpu_count() - 1) # Leave one core free
        print(f"--- Starting Training using {num_cpu} cores ---")
        
        # Create a unique log directory for this run
        # e.g., ./ppo_battlebots_tensorboard/ppo_gamma_0.98/
        run_log_dir = os.path.join(args.log_dir, os.path.splitext(args.model_name)[0])
        print(f"Logging to: {run_log_dir}")
        print(f"Saving model to: {args.model_name}")
        
        env = make_vec_env(BattleBotsEnv, n_envs=num_cpu,
                           vec_env_cls=SubprocVecEnv, env_kwargs=TRAINING_KWARGS)
        
        policy_kwargs = dict(net_arch=dict(pi=[256, 512, 256], vf=[256, 256]))
        
        model = PPO("MlpPolicy", env, verbose=1,
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=run_log_dir,  # Use the unique log directory
                    
                    # --- Tuned Hyperparameters ---
                    learning_rate=linear_schedule(args.lr),
                    n_steps=2048,
                    batch_size=64,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    ent_coef=args.ent_coef,
                    vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm,
                    
                    # device='cpu' # Force CPU, as it's often faster for MlpPolicy
                    )
        
        print(f"\n--- RUNNING WITH PARAMS ---")
        print(f"  lr: {args.lr}")
        print(f"  gamma: {args.gamma}")
        print(f"  gae_lambda: {args.gae_lambda}")
        print(f"  clip_range: {args.clip_range}")
        print(f"  ent_coef: {args.ent_coef}")
        print(f"  vf_coef: {args.vf_coef}")
        print(f"  max_grad_norm: {args.max_grad_norm}")
        print("-----------------------------\n")
        
        model.learn(total_timesteps=500_000, progress_bar=True)
        model.save(args.model_name) # Save to the specified model name
        
        print(f"--- Training Finished. Model saved to {args.model_name} ---")
        print("Set TRAIN_MODEL to False to watch the agent.")
    else:
        # --- Visualization Logic (unchanged) ---
        
        # Note: If you want to watch a *specific* tuned model,
        # you must change MODEL_FILENAME at the top of the script
        # to match the one you want to load.
        # Or, you could update this 'else' block to use args.model_name.
        
        watch_model = MODEL_FILENAME
        if args.model_name != MODEL_FILENAME:
            watch_model = args.model_name
            print(f"[Info] --model_name specified, overriding default.")
        
        print(f"--- Starting GUI Visualization (loading {watch_model}) ---")
        if not os.path.exists(watch_model):
            print(f"Error: Model file '{watch_model}' not found. Please train first.")
        else:
            model = PPO.load(watch_model)
            run_gui(model, env_kwargs=TESTING_KWARGS)
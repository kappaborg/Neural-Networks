import numpy as np
import pygame
import random
from enum import Enum

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    GOAL = 2
    TRAP = 3
    REWARD = 4
    TELEPORT = 5
    ICE = 6
    MUD = 7

class AdvancedGridWorld:
    def __init__(self, size=10, difficulty='medium'):
        self.size = size
        self.difficulty = difficulty
        self.reset()
        
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = 4
        self.observation_space = size * size * len(CellType)
        
        # Dynamic elements
        self.moving_obstacles = []
        self.teleport_pairs = []
        self.weather_condition = 'normal'
        self.time_of_day = 0  # 0-23 hours
        
        # Initialize special cells
        self._initialize_special_cells()
        
    def _initialize_special_cells(self):
        """Initialize special cells based on difficulty"""
        if self.difficulty == 'easy':
            self._add_special_cells(CellType.TRAP, 2)
            self._add_special_cells(CellType.REWARD, 3)
            self._add_moving_obstacles(1)
        elif self.difficulty == 'medium':
            self._add_special_cells(CellType.TRAP, 3)
            self._add_special_cells(CellType.REWARD, 2)
            self._add_special_cells(CellType.ICE, 2)
            self._add_moving_obstacles(2)
            self._add_teleport_pairs(1)
        else:  # hard
            self._add_special_cells(CellType.TRAP, 4)
            self._add_special_cells(CellType.REWARD, 1)
            self._add_special_cells(CellType.ICE, 3)
            self._add_special_cells(CellType.MUD, 3)
            self._add_moving_obstacles(3)
            self._add_teleport_pairs(2)
    
    def _add_special_cells(self, cell_type, count):
        """Add special cells to the grid"""
        added = 0
        while added < count:
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if self.grid[x][y] == CellType.EMPTY:
                self.grid[x][y] = cell_type
                added += 1
    
    def _add_moving_obstacles(self, count):
        """Add moving obstacles"""
        for _ in range(count):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            direction = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
            self.moving_obstacles.append({
                'pos': [x, y],
                'direction': direction,
                'pattern': 'patrol'  # or 'random'
            })
    
    def _add_teleport_pairs(self, count):
        """Add teleport pairs"""
        for _ in range(count):
            # Add entry point
            x1, y1 = random.randint(0, self.size-1), random.randint(0, self.size-1)
            while self.grid[x1][y1] != CellType.EMPTY:
                x1, y1 = random.randint(0, self.size-1), random.randint(0, self.size-1)
            
            # Add exit point
            x2, y2 = random.randint(0, self.size-1), random.randint(0, self.size-1)
            while self.grid[x2][y2] != CellType.EMPTY:
                x2, y2 = random.randint(0, self.size-1), random.randint(0, self.size-1)
            
            self.grid[x1][y1] = CellType.TELEPORT
            self.grid[x2][y2] = CellType.TELEPORT
            self.teleport_pairs.append(((x1,y1), (x2,y2)))
    
    def reset(self):
        """Reset the environment"""
        self.grid = [[CellType.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = CellType.GOAL
        
        self.moving_obstacles = []
        self.teleport_pairs = []
        self.steps_taken = 0
        self.collected_rewards = 0
        
        self._initialize_special_cells()
        return self._get_state()
    
    def step(self, action):
        """Take a step in the environment"""
        self.steps_taken += 1
        
        # Store old position
        old_pos = self.agent_pos.copy()
        
        # Move agent
        if action == 0:  # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1:  # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2:  # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        elif action == 3:  # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        
        # Update time and weather
        self._update_environment()
        
        # Handle special cells
        reward, done = self._handle_special_cells()
        
        # Move obstacles
        self._move_obstacles()
        
        # Check if agent hits moving obstacle
        if self._check_obstacle_collision():
            reward = -50
            done = True
        
        return self._get_state(), reward, done
    
    def _handle_special_cells(self):
        """Handle interaction with special cells"""
        current_cell = self.grid[self.agent_pos[0]][self.agent_pos[1]]
        reward = -0.1  # Small penalty for each step
        done = False
        
        if current_cell == CellType.GOAL:
            reward = 100
            done = True
        elif current_cell == CellType.TRAP:
            reward = -50
            done = True
        elif current_cell == CellType.REWARD:
            reward = 10
            self.collected_rewards += 1
            self.grid[self.agent_pos[0]][self.agent_pos[1]] = CellType.EMPTY
        elif current_cell == CellType.TELEPORT:
            self._handle_teleport()
        elif current_cell == CellType.ICE:
            self._handle_ice_movement()
        elif current_cell == CellType.MUD:
            reward = -0.3  # Extra penalty for moving through mud
            
        return reward, done
    
    def _handle_teleport(self):
        """Handle teleportation"""
        current_pos = (self.agent_pos[0], self.agent_pos[1])
        for entry, exit in self.teleport_pairs:
            if current_pos == entry:
                self.agent_pos = [exit[0], exit[1]]
                break
            elif current_pos == exit:
                self.agent_pos = [entry[0], entry[1]]
                break
    
    def _handle_ice_movement(self):
        """Handle sliding on ice"""
        # Continue moving in the same direction until hitting a wall or non-ice cell
        while True:
            next_pos = self.agent_pos.copy()
            if self.last_action == 0:  # up
                next_pos[0] -= 1
            elif self.last_action == 1:  # right
                next_pos[1] += 1
            elif self.last_action == 2:  # down
                next_pos[0] += 1
            elif self.last_action == 3:  # left
                next_pos[1] -= 1
                
            # Check if next position is valid
            if (next_pos[0] < 0 or next_pos[0] >= self.size or
                next_pos[1] < 0 or next_pos[1] >= self.size or
                self.grid[next_pos[0]][next_pos[1]] != CellType.ICE):
                break
                
            self.agent_pos = next_pos
    
    def _update_environment(self):
        """Update environmental conditions"""
        self.time_of_day = (self.time_of_day + 1) % 24
        
        # Update weather randomly
        if random.random() < 0.05:  # 5% chance to change weather
            self.weather_condition = random.choice(['normal', 'foggy', 'stormy'])
    
    def _move_obstacles(self):
        """Move dynamic obstacles"""
        for obstacle in self.moving_obstacles:
            if obstacle['pattern'] == 'patrol':
                # Move in current direction
                obstacle['pos'][0] += obstacle['direction'][0]
                obstacle['pos'][1] += obstacle['direction'][1]
                
                # Reverse direction if hitting boundary
                if (obstacle['pos'][0] < 0 or obstacle['pos'][0] >= self.size or
                    obstacle['pos'][1] < 0 or obstacle['pos'][1] >= self.size):
                    obstacle['direction'] = (-obstacle['direction'][0], -obstacle['direction'][1])
                    obstacle['pos'][0] += 2 * obstacle['direction'][0]
                    obstacle['pos'][1] += 2 * obstacle['direction'][1]
            else:  # random movement
                direction = random.choice([(0,1), (1,0), (0,-1), (-1,0)])
                new_pos = [
                    obstacle['pos'][0] + direction[0],
                    obstacle['pos'][1] + direction[1]
                ]
                if (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
                    obstacle['pos'] = new_pos
    
    def _check_obstacle_collision(self):
        """Check if agent collides with any moving obstacle"""
        return any(self.agent_pos == obstacle['pos'] for obstacle in self.moving_obstacles)
    
    def _get_state(self):
        """Get current state representation"""
        # Create a channel for each cell type
        state = np.zeros((len(CellType), self.size, self.size))
        
        # Fill in the grid
        for i in range(self.size):
            for j in range(self.size):
                cell_type = self.grid[i][j]
                state[cell_type.value][i][j] = 1
        
        # Add agent position
        agent_channel = np.zeros((self.size, self.size))
        agent_channel[self.agent_pos[0]][self.agent_pos[1]] = 1
        state = np.vstack([state, agent_channel.reshape(1, self.size, self.size)])
        
        # Add environmental conditions
        state = np.append(state.flatten(), [
            self.time_of_day / 24.0,
            1.0 if self.weather_condition == 'foggy' else 0.0,
            1.0 if self.weather_condition == 'stormy' else 0.0
        ])
        
        return state
    
    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            for i in range(self.size):
                for j in range(self.size):
                    if [i, j] == self.agent_pos:
                        print('A', end=' ')
                    elif [i, j] == self.goal_pos:
                        print('G', end=' ')
                    else:
                        cell_type = self.grid[i][j]
                        if cell_type == CellType.WALL:
                            print('W', end=' ')
                        elif cell_type == CellType.TRAP:
                            print('T', end=' ')
                        elif cell_type == CellType.REWARD:
                            print('R', end=' ')
                        elif cell_type == CellType.TELEPORT:
                            print('P', end=' ')
                        elif cell_type == CellType.ICE:
                            print('I', end=' ')
                        elif cell_type == CellType.MUD:
                            print('M', end=' ')
                        else:
                            print('.', end=' ')
                print()
            print(f"Steps: {self.steps_taken}, Rewards: {self.collected_rewards}")
            print(f"Time: {self.time_of_day}:00, Weather: {self.weather_condition}")
            print("------------------------") 
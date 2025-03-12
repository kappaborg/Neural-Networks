import numpy as np

class GridWorldEnv:
    def __init__(self, size=10):
        self.size = size
        self.reset()
        
        # Action space: 0: up, 1: right, 2: down, 3: left
        self.action_space = 4
        self.observation_space = size * size
        
    def reset(self):
        """Reset the environment"""
        self.grid = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size-1, self.size-1]
        return self._get_state()
    
    def step(self, action):
        """Take a step in the environment"""
        # Initialize reward and done flag
        reward = -0.1  # Small penalty for each step
        done = False
        
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
            
        # Check if goal is reached
        if self.agent_pos == self.goal_pos:
            reward = 100
            done = True
            
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Convert current position to state representation"""
        state = np.zeros(self.size * self.size)
        state[self.agent_pos[0] * self.size + self.agent_pos[1]] = 1
        return state
    
    def render(self):
        """Print the current state of the grid"""
        for i in range(self.size):
            for j in range(self.size):
                if [i, j] == self.agent_pos:
                    print('A', end=' ')
                elif [i, j] == self.goal_pos:
                    print('G', end=' ')
                else:
                    print('.', end=' ')
            print()
        print("------------------------") 
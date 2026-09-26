import numpy as np
import torch

class ReplayBuffer():

    def __init__(self, max_size, input_shape, device='cpu'):
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.next_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.uint8)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.device = device

    def can_sample(self, batch_size, min_buffer_size=10000):
        return self.mem_ctr >= max(batch_size, min_buffer_size)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_ctr % self.mem_size

        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = torch.tensor(self.state_memory[batch], dtype=torch.uint8).to(self.device)
        next_states = torch.tensor(self.next_state_memory[batch], dtype=torch.uint8).to(self.device)
        actions = torch.tensor(self.action_memory[batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor(self.reward_memory[batch], dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.terminal_memory[batch], dtype=torch.bool).to(self.device)

        return states, actions, rewards, next_states, dones

        

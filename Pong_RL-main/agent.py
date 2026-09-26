from buffer import ReplayBuffer
from model import Model
import torch
import torch.optim as optim
import torch.nn.functional as F
import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import random
import os
import cv2


class Agent():

    def __init__(self, env, hidden_layer=512, learning_rate=0.0001, gamma=0.99, step_repeat=1):
        self.env = env
        self.gamma = gamma 
        self.step_repeat = step_repeat
        
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("Using device:", self.device)
        
        self.memory = ReplayBuffer(max_size=100000, input_shape=obs.shape, device=self.device)
        self.model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model = Model(action_dim=self.env.action_space.n, hidden_dim=hidden_layer, observation_shape=obs.shape).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # ---> ระบบโหลดโมเดลเก่ามาต่อยอดอัตโนมัติ <---
        model_path = 'models/latest.pt'
        self.start_episode = 0
        self.saved_epsilon = None
        if os.path.exists(model_path):
            print(f"--> พบโมเดลเก่าที่ {model_path} กำลังโหลด...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.target_model.load_state_dict(self.model.state_dict())
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    self.start_episode = checkpoint.get('episode', 0) + 1
                    self.saved_epsilon = checkpoint.get('epsilon', None)
                    print(f"--> โหลดสำเร็จ! เทรนต่อจาก Episode ที่ {self.start_episode}")
                else:
                    self.model.load_state_dict(checkpoint)
                    self.target_model.load_state_dict(self.model.state_dict())
                    print("--> โหลดน้ำหนักเดิมสำเร็จ")
            except Exception as e:
                print(f"--> ไม่สามารถโหลด checkpoint เก่าได้เนื่องจากโครงสร้างโมเดลเปลี่ยนไป: {e}")
                print("--> เริ่มเทรนใหม่ตั้งแต่ต้น (Fresh Start)")
        else:
            print("--> ไม่พบโมเดลเก่า เริ่มเทรนใหม่ตั้งแต่ต้น (Fresh Start)")

    def process_observation(self, obs):
        # obs is numpy array of shape (4, 84, 84) from FrameStackObservation
        return obs 

    def test(self, model_path='models/latest.pt'):
        print(f"กำลังโหลดโมเดลจาก: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        obs, info = self.env.reset()
        obs = self.process_observation(obs)

        done = False
        episode_reward = 0

        while not done:
            if random.random() < 0.02:
                action = self.env.action_space.sample()
            else:
                state_tensor = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.model(state_tensor)
                action = torch.argmax(q_values, dim=-1).item()

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            try:
                frame = self.env.render()
                if frame is not None:
                    resized_frame = cv2.resize(frame, (500, 400))
                    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Pong AI Test", resized_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception:
                pass

            obs = self.process_observation(next_obs)
            episode_reward += reward

        cv2.destroyAllWindows()
        print(f"Test Finished! Episode Reward: {episode_reward}")

    def train(self, episodes, max_episode_steps, summary_writer_suffix, batch_size=32, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.05, target_update_freq=1000, min_buffer_size=10000):
        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{summary_writer_suffix}'
        writer = SummaryWriter(summary_writer_name)

        if not os.path.exists('models'):
            os.makedirs('models')

        # Restore epsilon if saved, or calculate from start_episode
        if self.saved_epsilon is not None:
            epsilon = self.saved_epsilon
            print(f"--> โหลดค่า Epsilon เดิม: {epsilon:.4f}")
        elif self.start_episode > 0:
            epsilon = max(min_epsilon, epsilon * (epsilon_decay ** self.start_episode))
            print(f"--> คำนวณค่า Epsilon ต่อจาก Episode {self.start_episode}: {epsilon:.4f}")

        total_steps = 0

        print(f"Starting training... Buffer Warmup requirement: {min_buffer_size} steps")

        for episode in range(self.start_episode, self.start_episode + episodes):
            done = False
            episode_reward = 0
            obs, info = self.env.reset()
            obs = self.process_observation(obs)

            episode_steps = 0
            episode_start_time = time.time()

            while not done and episode_steps < max_episode_steps:
                if random.random() < epsilon:
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                    action = torch.argmax(q_values, dim=-1).item()

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                next_obs = self.process_observation(next_obs)
                self.memory.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs

                episode_reward += reward
                episode_steps += 1
                total_steps += 1

                # Train model after replay buffer has enough samples
                if self.memory.can_sample(batch_size, min_buffer_size=min_buffer_size):
                    observations, actions, rewards, next_observations, dones = self.memory.sample_buffer(batch_size)

                    # Current Q-values
                    q_values = self.model(observations)
                    actions_tensor = actions.unsqueeze(1)
                    qsa_batch = q_values.gather(1, actions_tensor)

                    # Double DQN Target Calculation
                    with torch.no_grad():
                        next_actions = torch.argmax(self.model(next_observations), dim=1, keepdim=True)
                        next_q_values = self.target_model(next_observations).gather(1, next_actions)
                        target_b = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1).float()) * self.gamma * next_q_values

                    # Smooth L1 (Huber) Loss
                    loss = F.smooth_l1_loss(qsa_batch, target_b)

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                    self.optimizer.step()

                    writer.add_scalar("Loss/model", loss.item(), total_steps)

                    # Hard update target model
                    if total_steps % target_update_freq == 0:
                        self.target_model.load_state_dict(self.model.state_dict())

            # Epsilon decay per episode
            if epsilon > min_epsilon:
                epsilon = max(min_epsilon, epsilon * epsilon_decay)

            self.save_checkpoint(episode, epsilon)

            writer.add_scalar("Score", episode_reward, episode)
            writer.add_scalar("Epsilon", epsilon, episode)

            episode_time = time.time() - episode_start_time

            print(f"Episode {episode:5d} | Score: {episode_reward:5.1f} | Epsilon: {epsilon:.3f} | Steps: {episode_steps:4d} | Buffer: {self.memory.mem_ctr:6d} | Time: {episode_time:.2f}s")

    def save_checkpoint(self, episode, epsilon=1.0):
        if not os.path.exists('models'):
            os.makedirs('models')
            
        checkpoint = {
            'episode': episode,
            'epsilon': epsilon,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, 'models/latest.pt')
        
        if episode % 100 == 0:
            torch.save(checkpoint, f'models/model_ep{episode}.pt')
            print(f"--> สำรอง Checkpoint ที่ Episode {episode} เรียบร้อย")
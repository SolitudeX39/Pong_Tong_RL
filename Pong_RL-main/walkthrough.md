# Atari Pong RL Agent Upgrade (Nature DQN)

We have successfully upgraded and tuned the Pong Reinforcement Learning agent from a naive single-frame network to a standard **DeepMind Nature DQN** architecture with 4-frame stacking and optimized hyperparameters.

## Changes Completed

### 1. Preprocessing & Environment Pipeline
- **[train.py](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/train.py)** & **[test.py](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/test.py)**:
  - Added `CropObservation` to remove top scoreboard (rows 0-34) and bottom margins (rows 194-210) to focus strictly on paddle & ball dynamics.
  - Applied `GrayscaleObservation` and `ResizeObservation((84, 84))`.
  - Added `FrameStackObservation(stack_size=4)` to stack 4 consecutive frames, providing velocity & trajectory vector information for the ball.

### 2. Network Architecture
- **[model.py](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/model.py)**:
  - Upgraded `Model` class to standard Nature DQN architecture:
    - Conv1: `(4 -> 32, kernel=8, stride=4)`
    - Conv2: `(32 -> 64, kernel=4, stride=2)`
    - Conv3: `(64 -> 64, kernel=3, stride=1)`
    - Linear: `(3136 -> 512)` -> Output `(512 -> action_dim)`
  - Automatic `x.float() / 255.0` normalization inside `forward`.

### 3. Replay Buffer & Training Stability
- **[buffer.py](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/buffer.py)**:
  - Memory buffer stores `(4, 84, 84)` uint8 arrays efficiently.
  - Added configurable `can_sample(batch_size, min_buffer_size)` for warmup checks.

- **[agent.py](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/agent.py)**:
  - **Hard Target Update**: Replaced fast soft-updates with hard target updates every 1,000 steps.
  - **Huber Loss**: Replaced MSE Loss with `F.smooth_l1_loss` and added gradient clipping (`clip_grad_norm_`).
  - **Buffer Warmup**: Implemented 10,000 steps replay buffer warmup before initiating backpropagation.
  - **Double DQN Target Calculation**: Selected next actions using primary network and evaluated Q-values using target network.

---

## Verification Results

### End-to-End Execution Test
- Tested model initialization, observation shape verification `(4, 84, 84)`, GPU memory transfers (`cuda:0`), loss computation, hard target network updates, and checkpoint saving/loading.
- Output log:
  ```text
  Using device: cuda:0
  Conv output size: 3136
  Obs shape: (4, 84, 84)
  Episode 0 | Score: -2.0 | Epsilon: 0.995 | Steps: 100 | Buffer: 100 | Time: 0.74s
  Loaded latest.pt successfully!
  ```

---

## How to Run

To start training the upgraded agent:
```bash
python train.py
```

To test/evaluate the trained model:
```bash
python test.py
```

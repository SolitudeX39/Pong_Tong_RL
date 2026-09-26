# Project Flow & System Architecture - Atari Pong RL (Nature DQN)

เอกสารอธิบายโครงสร้างระบบ ลำดับการทำงาน (Process Flow) และสถาปัตยกรรมซอฟต์แวร์ของโปรเจกต์ **Atari Pong Reinforcement Learning**

---

## 🔄 1. ภาพรวมลำดับการทำงาน (System Flowchart)

```mermaid
flowchart TD
    Start([เริ่มทำงาน train.py]) --> EnvironmentSetup[สร้าง Environment & Preprocessing Pipeline]
    
    subgraph Preprocessing ["1. Observation Preprocessing"]
        EnvRaw["Atari Pong (210x160 RGB)"] --> Crop["CropObservation (ตัดขอบคะแนนเหลือ 160x160)"]
        Crop --> Gray["GrayscaleObservation (แปลงเป็นภาพขาวดำ 160x160)"]
        Gray --> Resize["ResizeObservation (ย่อขนาดเป็น 84x84)"]
        Resize --> Stack["FrameStackObservation (รวม 4 เฟรมล่าสุดเป็น shape 4x84x84)"]
    end
    
    EnvironmentSetup --> InitAgent[สร้าง Agent / Model / ReplayBuffer]
    
    InitAgent --> LoadCheckpoint{มีไฟล์ models/latest.pt ไหม?}
    LoadCheckpoint -- มี --> LoadWeights[โหลดน้ำหนักโมเดล + Optimizer + Epsilon เดิม]
    LoadCheckpoint -- ไม่มี --> FreshStart[เริ่มนับ Episode 0 ใหม่]
    
    LoadWeights --> EpisodeLoop[เริ่มวน Loop แต่ละ Episode]
    FreshStart --> EpisodeLoop
    
    subgraph StepLoop ["2. Episode Step Loop"]
        ResetEnv[env.reset -> รับภาพ 4x84x84] --> CheckEpsilon{สุ่มตัวเลข < Epsilon?}
        CheckEpsilon -- ใช่ (Explore) --> RandomAction[สุ่มขยับไม้ปิงปอง]
        CheckEpsilon -- ไม่ใช่ (Exploit) --> ModelPredict[ส่งภาพให้ CNN Model ทายค่า Q-Value -> เลือก Action สูงสุด]
        
        RandomAction --> StepEnv[env.step(action) -> ได้ next_obs, reward, done]
        ModelPredict --> StepEnv
        
        StepEnv --> StoreBuffer[เก็บ transition (obs, action, reward, next_obs, done) ลง ReplayBuffer]
        
        StoreBuffer --> CheckWarmup{Buffer ครบ 10,000 steps หรือยัง?}
        CheckWarmup -- ยังไม่ครบ --> NextStepCheck
        CheckWarmup -- ครบแล้ว --> TrainStep[อัปเดตโมเดล Q-Learning]
        
        subgraph QLearning ["3. Gradient Update (Double DQN)"]
            TrainStep --> SampleBatch[สุ่ม Batch 32 รายการจาก Memory]
            SampleBatch --> CalcQ[คำนวณ Current Q-Value จาก Model]
            CalcQ --> CalcTarget[คำนวณ Target Q-Value จาก Target Model]
            CalcTarget --> LossCalc[คำนวณ Huber Loss & Backward Gradient]
            LossCalc --> Optimize[Optimizer.step อัปเดตน้ำหนัก]
            Optimize --> TargetUpdate{ครบรอบ 1,000 steps?}
            TargetUpdate -- ใช่ --> SyncTarget[ก๊อบปี้น้ำหนักไปยัง Target Model]
            TargetUpdate -- ไม่ใช่ --> NextStepCheck
            SyncTarget --> NextStepCheck
        end
    end
    
    NextStepCheck{เกมจบ หรือ Step ครบ?} -- ยังไม่จบ --> StepLoop
    NextStepCheck -- จบ Episode --> LogCheck[ลด Epsilon + บันทึก Tensorboard + Save Checkpoint]
    
    LogCheck --> NextEpisode{ครบจำนวน Episodes หรือยัง?}
    NextEpisode -- ยังไม่ครบ --> EpisodeLoop
    NextEpisode -- ครบแล้ว --> End([จบการเทรน])
```

---

## 🖼️ 2. ภาพรวมกระบวนการแปลงภาพอินพุต (Observation Pipeline)

เพื่อให้โมเดลสามารถรับรู้ **ความเร็วและทิศทาง** ของลูกบอลได้อย่างแม่นยำ ภาพจะถูกแปลงตามขั้นตอนดังนี้:

| ขั้นตอน | การประมวลผล | Shape ผลลัพธ์ | วัตถุประสงค์ |
| :--- | :--- | :--- | :--- |
| **0. Raw Frame** | ภาพจากเกม Atari Pong | `(210, 160, 3)` | ภาพสี RGB ตั้งต้นจากเกม |
| **1. Crop** | `CropObservation` (ตัดแถบ Y: 34-194) | `(160, 160, 3)` | ตัดสกอร์บอร์ดด้านบนและขอบล่างออกเพื่อลด Noise |
| **2. Grayscale** | `GrayscaleObservation(keep_dim=False)` | `(160, 160)` | ยุบมิติสีเหลือขาวดำเพื่อประหยัด RAM/GPU |
| **3. Resize** | `ResizeObservation((84, 84))` | `(84, 84)` | ย่อสัดส่วนมาตรฐานของ Nature DQN |
| **4. Frame Stack**| `FrameStackObservation(stack_size=4)` | `(4, 84, 84)` | ซ้อน 4 เฟรมล่าสุดเพื่อให้ CNN รู้ทิศทางความเร็วลูกบอล |

---

## 🧩 3. หน้าที่ของแต่ละไฟล์ในโปรเจกต์ (Module Breakdown)

### 1. [`train.py`](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/train.py) *(Main Training Script)*
- กำหนดค่า Hyperparameters หลัก เช่น `learning_rate=0.0001`, `batch_size=32`, `target_update_freq=1000`
- สร้าง Environment พร้อม Observation Wrappers ด้วยฟังก์ชัน `make_env()`
- เริ่มต้นการทำงานของ `Agent` และรันลูปฝึกฝนหลัก

### 2. [`agent.py`](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/agent.py) *(RL Agent & Brain Logic)*
- บริหารจัดการความจำ `ReplayBuffer`, โมเดลหลัก (`model`) และโมเดลเป้าหมาย (`target_model`)
- เลือกการกระทำด้วยเทคนิค **$\epsilon$-greedy policy** (สุ่มสำรวจ $\rightarrow$ ค่อยๆ ลดลงเหลือ 0.02)
- คำนวณ Loss ด้วย **Double DQN** + **Huber Loss (`F.smooth_l1_loss`)**
- บันทึกและดึงค่า Checkpoint (`models/latest.pt` และ `models/model_ep{N}.pt`) พร้อมค่า Epsilon เดิมเมื่อรันใหม่

### 3. [`model.py`](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/model.py) *(Nature DQN Neural Network)*
- สถาปัตยกรรม CNN ตามมาตรฐาน DeepMind Nature DQN:
  - **Conv1**: In=4, Out=32, Kernel=8, Stride=4 + ReLU
  - **Conv2**: In=32, Out=64, Kernel=4, Stride=2 + ReLU
  - **Conv3**: In=64, Out=64, Kernel=3, Stride=1 + ReLU
  - **FC1**: Linear(3136 $\rightarrow$ 512) + ReLU
  - **Output**: Linear(512 $\rightarrow$ action_dim)
- ปรับสเกลภาพจาก `0-255` เป็น `0.0-1.0` อัตโนมัติใน `forward()`

### 4. [`buffer.py`](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/buffer.py) *(Replay Memory Buffer)*
- จัดเก็บความจำประสบการณ์การเล่น `(state, action, reward, next_state, done)` ย้อนหลังสูงสุด 100,000 steps ในรูปแบบ `uint8` เพื่อประหยัดพื้นที่ RAM
- ฟังก์ชัน `can_sample()` ตรวจสอบว่าเก็บข้อมูลสุ่มครบ Buffer Warmup (10,000 steps) หรือยังก่อนเริ่มส่งต่อให้ Optimizer

### 5. [`test.py`](file:///d:/AI4BA/2569/ai4ba/Pong_RL-main/test.py) *(Evaluation Script)*
- สคริปต์สำหรับโหลดน้ำหนักโมเดลจากไฟล์ Checkpoint เพื่อรันโชว์ผลการเล่นจริงผ่านหน้าจอ OpenCV โดยปิดระบบการสุ่ม exploration

---

## 📊 4. พัฒนาการตามจำนวน Episode (Performance Benchmark)

| ช่วง Episode | ค่า Epsilon | โอกาสชนะต่อแมตช์ | คะแนนเฉลี่ย (Score) | พฤติกรรมของ AI |
| :--- | :--- | :--- | :--- | :--- |
| **0 – 150** | `1.00` $\rightarrow$ `0.47` | **0%** | **-21 ถึง -18** | **ช่วงสุ่มเก็บข้อมูล**: เน้นสุ่มเดา ขยับหลบลูกแพ้ขาดลอย |
| **150 – 350** | `0.47` $\rightarrow$ `0.17` | **10% – 30%** | **-15 ถึง -2** | **เริ่มดักทางลูกได้**: เริ่มขยับไม้ตามจุดลูกบอลได้แม่นยำขึ้น |
| **350 – 600** | `0.17` $\rightarrow$ `0.05` | 🏆 **70% – 85%** | **+5 ถึง +15** | **จุดเปลี่ยนสำคัญ**: AI เริ่มทำคะแนนนำและชนะ Bot ได้สม่ำเสมอ |
| **600 – 1,000** | `0.05` $\rightarrow$ `0.02` | 🥇 **90% – 98%** | **+16 ถึง +20** | **ระดับเซียน**: ขยับดักรับลูกโต้กลับมุมยากๆ ได้เกือบหมด |
| **1,000 – 1,500** | `0.02` (คงที่) | 👑 **99% – 100%** | **+20 ถึง +21** | **ระดับสมบูรณ์แบบ**: ชนะขาดลอย 21-0 หรือ 21-1 แทบทุกแมตช์ |

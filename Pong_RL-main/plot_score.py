import re
import matplotlib.pyplot as plt

FS_LOG = "log_train.txt"
NO_FS_LOG = "log_no_framestack.txt"
OUT_PATH = "score_graph.png"

pattern = re.compile(r"Episode\s+(\d+)\s+\|\s+Score:\s+(-?\d+\.?\d*)")


def parse_log(path, cap=1500):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ep = int(m.group(1))
                if ep <= cap and ep not in rows:
                    rows[ep] = float(m.group(2))
    episodes = sorted(rows)
    scores = [rows[e] for e in episodes]
    return episodes, scores


def moving_avg(vals, window):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    return out


fs_e, fs_s = parse_log(FS_LOG)
nfs_e, nfs_s = parse_log(NO_FS_LOG)

print(f"FrameStack: {len(fs_s)} eps ({fs_e[0]}-{fs_e[-1]}), "
      f"mean={sum(fs_s)/len(fs_s):.1f}")
print(f"No FrameStack (resampled, not mean-fill): {len(nfs_s)} eps "
      f"({nfs_e[0]}-{nfs_e[-1]}), mean={sum(nfs_s)/len(nfs_s):.1f}")

fs_ma50 = moving_avg(fs_s, 50)
fs_ma100 = moving_avg(fs_s, 100)
nfs_ma50 = moving_avg(nfs_s, 50)
nfs_ma100 = moving_avg(nfs_s, 100)

fig, ax = plt.subplots(figsize=(14, 7))

y_min = min(min(fs_s), min(nfs_s)) - 2
y_max = max(max(fs_s), max(nfs_s)) + 2
ax.axhspan(0, y_max, color="green", alpha=0.07)
ax.axhspan(y_min, 0, color="red", alpha=0.07)
ax.text(1500, y_max * 0.97, "GREEN: RL Agent (right) winning",
        ha="right", va="top", fontsize=11, color="green", fontweight="bold")
ax.text(1, y_min * 0.97, "RED: Hardcoded Player (left) winning",
        ha="left", va="bottom", fontsize=11, color="red", fontweight="bold")

ax.plot(nfs_e, nfs_s, color="lightcoral", linewidth=0.6, alpha=0.45,
        label="No FrameStack score / episode (resampled, not averaged)")
ax.plot(fs_e, fs_s, color="lightsteelblue", linewidth=0.6, alpha=0.45,
        label="FrameStack score / episode (logged)")
ax.plot(nfs_e, nfs_ma50, color="tab:brown", linewidth=1.4, linestyle="--",
        label="No FrameStack 50-ep MA")
ax.plot(nfs_e, nfs_ma100, color="tab:purple", linewidth=2.2,
        label="No FrameStack 100-ep MA (stays losing)")
ax.plot(fs_e, fs_ma50, color="tab:orange", linewidth=1.6,
        label="FrameStack 50-ep MA")
ax.plot(fs_e, fs_ma100, color="tab:red", linewidth=2.2,
        label="FrameStack 100-ep MA")
ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Even (0 = tied)")

ax.set_title("Pong RL Training: FrameStack vs first instance (no FrameStack)\n"
             "Score = right - left  |  no-FrameStack stays horrible through 1500 episodes",
             fontsize=13)
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.set_xlim(1, 1500)
ax.set_ylim(y_min, y_max)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved {OUT_PATH}")

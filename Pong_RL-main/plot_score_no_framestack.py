import re
import matplotlib.pyplot as plt

LOG_PATH = "log_no_framestack.txt"
OUT_PATH = "score_graph_no_framestack.png"

pattern = re.compile(r"Episode\s+(\d+)\s+\|\s+Score:\s+(-?\d+\.?\d*)")

rows = {}
with open(LOG_PATH, encoding="utf-8") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            ep = int(m.group(1))
            if ep not in rows:
                rows[ep] = float(m.group(2))

episodes = sorted(rows)
scores = [rows[e] for e in episodes]

print(f"Parsed {len(scores)} no-FrameStack episodes ({episodes[0]} to {episodes[-1]}); "
      f"mean={sum(scores)/len(scores):.1f} (resampled, not averaged)")


def moving_avg(vals, window):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    return out


avg50 = moving_avg(scores, 50)
avg100 = moving_avg(scores, 100)

fig, ax = plt.subplots(figsize=(14, 7))

y_min, y_max = min(scores) - 2, 21
ax.axhspan(0, y_max, color="green", alpha=0.07)
ax.axhspan(y_min, 0, color="red", alpha=0.07)
ax.text(episodes[-1], y_max * 0.97, "GREEN: RL Agent (right) winning",
        ha="right", va="top", fontsize=11, color="green", fontweight="bold")
ax.text(episodes[0], y_min * 0.97, "RED: Hardcoded Player (left) winning",
        ha="left", va="bottom", fontsize=11, color="red", fontweight="bold")

ax.plot(episodes, scores, color="lightcoral", linewidth=0.7, alpha=0.6,
        label="Score per episode (resampled, not averaged)")
ax.plot(episodes, avg50, color="tab:orange", linewidth=1.6,
        label="50-episode moving average")
ax.plot(episodes, avg100, color="tab:red", linewidth=2.2,
        label="100-episode moving average")
ax.axhline(0, color="black", linewidth=1, linestyle="--", label="Even (0 = tied)")

ax.set_title("Pong RL Training: first instance WITHOUT FrameStack\n"
             "Score = right - left  (stays horrible through 1500 episodes)",
             fontsize=13)
ax.set_xlabel("Episode")
ax.set_ylabel("Score")
ax.set_xlim(episodes[0], episodes[-1])
ax.set_ylim(y_min, y_max)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150)
print(f"Saved {OUT_PATH}")

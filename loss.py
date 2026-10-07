import re
import pandas as pd
import matplotlib.pyplot as plt

RICOTTA_LOG = 'runs/ricotta-2.0/r1-wsd10(minlr0.1)/ricotta_2_0/train.log'
GRUYERE_LOG = 'runs/gruyere-2.0/r1/gruyere_2_0/train.log'
GOUDA_LOG = 'runs/gouda-2.0/r1/gouda_2_0/train.log'
CHEDDAR_LOG = 'runs/cheddar-2.0/r1/cheddar_2_0/train.log'
OUTPUT_FILE = "loss_comparison.png"

SMOOTHING_WINDOW = 100

Y_MIN = 3.0
Y_MAX = 4.0


LOGS = {
    "Ricotta-2.0": RICOTTA_LOG,
    "Gruyère-2.0": GRUYERE_LOG,
    "Gouda-2.0": GOUDA_LOG,
    "Cheddar-2.0": CHEDDAR_LOG
}

def parse_log(log_file):
    train_rows = []
    val_rows = []

    with open(log_file, "r") as f:
        for line in f:
            parts = line.split()

            if len(parts) < 2 or not parts[0].isdigit():
                continue

            step = int(parts[0])
            kind = parts[1]

            values = {
                k: float(v)
                for k, v in re.findall(
                    r"(\w+)=([\d.eE+-]+)",
                    line
                )
            }

            if kind == "train":
                train_rows.append({"step": step,**values,})

            elif kind == "val":
                val_rows.append({"step": step,**values,})

    train = pd.DataFrame(train_rows)
    val = pd.DataFrame(val_rows)

    return train, val

data = {}


for name, path in LOGS.items():
    print(f"Loading {name}: {path}")

    train, val = parse_log(path)

    print(f"  Train points: {len(train):,} | " f"Validation points: {len(val):,}")

    if len(train):
        print(f"  Tokens: " f"{train['tokens'].min() / 1e9:.3f}B -> " f"{train['tokens'].max() / 1e9:.3f}B")

    data[name] = (train, val)



fig, ax = plt.subplots(figsize=(12, 6))

for name, (train, val) in data.items():

    if train.empty:
        print(f"Warning: no training data found for {name}")
        continue

    if "loss" not in train.columns:
        print(f"Warning: no loss field found for {name}")
        continue

    if "tokens" not in train.columns:
        print(f"Warning: no tokens field found for {name}")
        continue

    # Rolling average of training loss
    smooth = train["loss"].rolling(window=SMOOTHING_WINDOW, min_periods=1,).mean()

    # Training curve
    line, = ax.plot(train["tokens"] / 1e9, smooth, linewidth=2, label=f"{name} train",)

    # Validation curve
    if (not val.empty and "loss" in val.columns and "tokens" in val.columns):
        ax.plot(val["tokens"] / 1e9, val["loss"], marker="o", markersize=4, linewidth=1.5, linestyle="--", color=line.get_color(), label=f"{name} val",)


ax.set_xlabel("Training Tokens (billions)")
ax.set_ylabel("Cross-Entropy Loss")

ax.set_title("Gouda Architecture 2.0 — Loss vs Training Tokens")
ax.set_ylim(bottom=2.5, top=Y_MAX,)
ax.grid(True, alpha=0.3,)
ax.legend()
fig.tight_layout()


fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight",)
plt.close(fig)

print(f"\nSaved graph to: {OUTPUT_FILE}")
"""Plot a histogram of label frequencies from a Raven selection table."""

import pandas as pd
import matplotlib.pyplot as plt

LABEL_FILE = "ryjo_labels/MARS_20161221_000046_SongSession_16kHz_HPF5HzNorm_labels.txt"

df = pd.read_csv(LABEL_FILE, sep="\t")
label_col = "Type"

counts = df[label_col].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(counts.index, counts.values, edgecolor="black", linewidth=0.6)

# Color bars with a nice palette
cmap = plt.cm.Set2
for i, bar in enumerate(bars):
    bar.set_color(cmap(i / max(len(bars) - 1, 1)))

ax.set_xlabel("Label", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.set_title(
    "Label Frequency — MARS_20161221_000046_SongSession",
    fontsize=15,
    fontweight="bold",
)

# Annotate bars with counts
for bar, count in zip(bars, counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(count),
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=11)
plt.tight_layout()
plt.savefig("scripts/label_histogram.png", dpi=150)
plt.show()
print("\nLabel counts:\n", counts.to_string())

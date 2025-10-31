import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_environment(safe_set, obstacle, target):
    def plot_rect(ax, rect, facecolor, edgecolor='black', label=None, textcolor='black'):
        lower, upper = rect.lower[0:2], rect.upper[0:2]
        width, height = (upper - lower).tolist()
        r = Rectangle(lower.tolist(), width, height,
                      facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
        ax.add_patch(r)
        if label:
            cx = (lower[0] + upper[0]) / 2
            cy = (lower[1] + upper[1]) / 2
            ax.text(cx, cy, label, color=textcolor, ha='center', va='center',
                    fontsize=12, fontweight='bold')

    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot sets
    plot_rect(ax, safe_set, facecolor='white', edgecolor='black')
    plot_rect(ax, obstacle, facecolor='black', edgecolor='black', label='obs', textcolor='white')
    plot_rect(ax, target, facecolor='gray', edgecolor='black', label='tgt', textcolor='black')

    lower = safe_set.lower[0:2]
    upper = safe_set.upper[0:2]
    ax.set_xlim(lower[0] - 0.5, upper[0] + 0.5)
    ax.set_ylim(lower[1] - 0.5, upper[1] + 0.5)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.show()

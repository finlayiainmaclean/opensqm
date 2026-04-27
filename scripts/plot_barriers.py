"""Module for plotting energy barriers."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add the project root to the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from opensqm.torsion_scanner import get_barrier_kcal  # noqa: E402


def main():
    """Run the main plot generation script."""
    half_lives = {
        "1 picosecond": 1e-12,
        "1 nanosecond": 1e-9,
        "1 microsecond": 1e-6,
        "1 millisecond": 1e-3,
        "1 second": 1,
        "1 minute": 60,
        "1 hour": 3600,
        "1 day": 86400,
        "1 month": 30 * 86400,
        "1 year": 365 * 86400,
    }

    # Generate a smooth curve for half-lives from 1 ms to 10 years
    t_min = 1e-3
    t_max = 31536000 * 10

    # Generate points in log scale
    t_points = np.logspace(np.log10(t_min), np.log10(t_max), 500)
    barriers = [get_barrier_kcal(t) for t in t_points]

    # Create the plot
    _fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the full line
    ax.plot(barriers, t_points, "k-", alpha=0.5, linewidth=2, zorder=1)

    # Plot dots and labels for specific timeframes
    colors = plt.cm.tab10(np.linspace(0, 1, len(half_lives)))

    for (label, t), color in zip(half_lives.items(), colors, strict=False):
        barrier = get_barrier_kcal(t)
        ax.scatter(
            barrier,
            t,
            color=color,
            s=120,
            zorder=5,
            edgecolors="black",
            label=f"{label} ({barrier:.1f} kcal/mol)",
        )

    # Format the y-axis to be logarithmic
    ax.set_yscale("log")

    # Set axis limits
    ax.set_xlim(min(barriers) - 1, max(barriers) + 1)

    # Labels and titles
    ax.set_ylabel("Half-life (seconds)")
    ax.set_xlabel("Barrier Height (kcal/mol)")
    ax.set_title("Half-life vs Kinetic Barrier at 300K", fontsize=14)

    # Add grid
    ax.grid(True, which="major", ls="-", alpha=0.2)
    ax.grid(True, which="minor", ls="--", alpha=0.1)

    # Add legend
    ax.legend(title="Timeframes", loc="lower right")

    # Save the plot
    out_path = Path(__file__).resolve().parent / "barrier_vs_halflife.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()

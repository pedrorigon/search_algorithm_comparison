import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import math

sns.set_theme(style="darkgrid")


def linear_search(arr, target):
    """
    Performs a Linear Search (O(n)) on the given array.

    Parameters:
    - arr (list): The array in which to search.
    - target (int): The value to be found.

    Returns:
    - (index, steps): A tuple where 'index' is the position of the target if found,
      otherwise -1, and 'steps' is the total number of iterations performed.
    """
    steps = 0
    for i in range(len(arr)):
        steps += 1
        if arr[i] == target:
            return i, steps
    return -1, steps


def jump_search(arr, target):
    """
    Performs a Jump Search (O(sqrt(n))) on the given array.
    The array must be sorted.

    Parameters:
    - arr (list): The sorted array in which to search.
    - target (int): The value to be found.

    Returns:
    - (index, steps): A tuple where 'index' is the position of the target if found,
      otherwise -1, and 'steps' is the total number of iterations performed.
    """

    steps = 0
    n = len(arr)
    jump = int(math.sqrt(n))
    prev = 0

    # Repeatedly jump by 'sqrt(n)' while the value at the jump index is less than target
    while prev < n and arr[min(jump, n) - 1] < target:
        steps += 1
        prev = jump
        jump += int(math.sqrt(n))
        if prev >= n:
            return -1, steps

    # Linear search within the identified block
    for i in range(prev, min(jump, n)):
        steps += 1
        if arr[i] == target:
            return i, steps
    return -1, steps


def binary_search(arr, target):
    """
    Performs a Binary Search (O(log n)) on the given array.
    The array must be sorted.

    Parameters:
    - arr (list): The sorted array in which to search.
    - target (int): The value to be found.

    Returns:
    - (index, steps): A tuple where 'index' is the position of the target if found,
      otherwise -1, and 'steps' is the total number of iterations performed.
    """
    head = 0
    tail = len(arr) - 1
    steps = 0
    while head <= tail:
        steps += 1
        mid = (head + tail) // 2
        if arr[mid] == target:
            return mid, steps
        elif arr[mid] < target:
            head = mid + 1
        else:
            tail = mid - 1
    return -1, steps


def run_search_algorithm(search_function):
    """
    Repeatedly runs the given search function for exponentially increasing list sizes
    (1, 2, 4, 8, ...). For each size, it calculates the average number of steps over
    60 random trials.

    Parameters:
    - search_function (callable): A function such as linear_search, jump_search, or binary_search.

    Returns:
    - (sizes, avg_steps): Two lists. 'sizes' holds the input sizes used, and 'avg_steps'
      contains the average steps for each input size.
    """
    results = []
    size = 1
    i = 0
    while i < 18:  # from 1 to 2^17
        steps_collection = []
        for _ in range(60):
            arr = random.sample(range(0, size * 2), size)
            arr.sort()
            # Random target from the array to ensure it exists
            target = random.choice(arr)
            _, steps = search_function(arr, target)
            steps_collection.append(steps)
        avg_steps = sum(steps_collection) / len(steps_collection)
        results.append([size, avg_steps])
        size *= 2
        i += 1
    return [item[0] for item in results], [item[1] for item in results]


def generate_all_plots_and_save():
    """
    Generate all figures, save each as PDF and PNG, and return the figure objects
    in a list for further usage (e.g., combining into a single PDF).
    """
    figs = []

    # Collect data from each algorithm
    binary_sizes, binary_steps = run_search_algorithm(binary_search)
    linear_sizes, linear_steps = run_search_algorithm(linear_search)
    jump_sizes, jump_steps = run_search_algorithm(jump_search)

    # Theoretical complexities
    log_n = np.log2(binary_sizes)
    sqrt_n = np.sqrt(jump_sizes)
    n_values = linear_sizes

    # Dictionary with display titles and data
    algorithms = {
        "Binary Search (O(log n))": (binary_sizes, binary_steps, log_n, "blue"),
        "Linear Search (O(n))": (linear_sizes, linear_steps, n_values, "orange"),
        "Jump Search (O($\\sqrt{n}$))": (jump_sizes, jump_steps, sqrt_n, "green"),
    }

    # ------------------------------------------------------------------------
    # 1) INDIVIDUAL PLOTS (2x2) FOR EACH ALGORITHM VS. THEORETICAL BASELINE
    #    -> NORMAL ASPECT RATIO
    # ------------------------------------------------------------------------
    fig_individual, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, (title, (sizes, steps, baseline, color)) in enumerate(algorithms.items()):
        ax = axes[idx]

        # Measured data
        ax.plot(
            sizes, steps, "o-", label="Measured", color=color, linewidth=2, markersize=5
        )
        # Theoretical line
        ax.plot(sizes, baseline, "--", label="Theoretical", color="red", linewidth=2)

        # Logarithmic scales
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("Input Size (n)", fontsize=10)
        ax.set_ylabel("Steps", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.set_xlim(left=sizes[0])
        ax.set_ylim(bottom=1)

        ax.grid(True, which="both", linestyle="--", linewidth=0.7)
        ax.legend(fontsize=9)

        # Annotation with a red arrow
        ax.annotate(
            f"{title}\nGrowth curve matches {title.split('(')[-1][:-1]} complexity.",
            xy=(sizes[-1], steps[-1]),
            xytext=(sizes[-1], steps[-1] * 3),
            arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
        )

    fig_individual.tight_layout()
    fig_individual.savefig("individual_plots.pdf", bbox_inches="tight")
    fig_individual.savefig("individual_plots.png", bbox_inches="tight")
    figs.append(fig_individual)

    # ------------------------------------------------------------------------
    # 2) COMPARISON PLOT (ALL ALGORITHMS)
    #    -> NORMAL ASPECT RATIO
    # ------------------------------------------------------------------------
    fig_all = plt.figure(figsize=(10, 6))
    plt.plot(
        linear_sizes,
        linear_steps,
        "o-",
        label="Linear Search (O(n))",
        color="orange",
        linewidth=2,
    )
    plt.plot(
        jump_sizes,
        jump_steps,
        "o-",
        label="Jump Search (O($\\sqrt{n}$))",
        color="green",
        linewidth=2,
    )
    plt.plot(
        binary_sizes,
        binary_steps,
        "o-",
        label="Binary Search (O(log n))",
        color="blue",
        linewidth=2,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1)
    plt.ylim(bottom=1)

    plt.xlabel("Input Size (n)", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.title(
        "Comparison: Linear, Jump, and Binary Search", fontsize=14, fontweight="bold"
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend(fontsize=10, loc="upper left")

    plt.annotate(
        "Linear Search (O(n)): grows proportionally to n.",
        xy=(linear_sizes[-1], linear_steps[-1]),
        xytext=(linear_sizes[-1], linear_steps[-1] * 0.3),
        arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
    )

    plt.annotate(
        "Jump Search (O($\\sqrt{n}$)): moderate growth,\noutperforms linear on large n.",
        xy=(jump_sizes[-1], jump_steps[-1]),
        xytext=(jump_sizes[-1], jump_steps[-1] * 1.5),
        arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
    )

    plt.annotate(
        "Binary Search (O(log n)): fastest among the three.",
        xy=(binary_sizes[-1], binary_steps[-1]),
        xytext=(binary_sizes[-1], binary_steps[-1] * 5),
        arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
    )

    fig_all.tight_layout()
    fig_all.savefig("comparison_all.pdf", bbox_inches="tight")
    fig_all.savefig("comparison_all.png", bbox_inches="tight")
    figs.append(fig_all)

    # ------------------------------------------------------------------------
    # 3) COMPARISON: JUMP VS. BINARY
    #    -> NORMAL ASPECT RATIO
    # ------------------------------------------------------------------------
    fig_jump_bin = plt.figure(figsize=(10, 6))
    plt.plot(
        jump_sizes,
        jump_steps,
        "o-",
        label="Jump Search (O($\\sqrt{n}$))",
        color="green",
        linewidth=2,
    )
    plt.plot(
        binary_sizes,
        binary_steps,
        "o-",
        label="Binary Search (O(log n))",
        color="blue",
        linewidth=2,
    )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(left=1)
    plt.ylim(bottom=1)

    plt.xlabel("Input Size (n)", fontsize=12)
    plt.ylabel("Steps", fontsize=12)
    plt.title("Comparison: Jump vs. Binary Search", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.legend(fontsize=10, loc="upper left")

    plt.annotate(
        "Jump Search (O($\\sqrt{n}$)): slower than binary,\nbut faster than linear.",
        xy=(jump_sizes[-1], jump_steps[-1]),
        xytext=(jump_sizes[-1], jump_steps[-1] * 3),
        arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
    )

    plt.annotate(
        "Binary Search (O(log n)): dominates\nfor large n.",
        xy=(binary_sizes[-1], binary_steps[-1]),
        xytext=(binary_sizes[-1], binary_steps[-1] * 6),
        arrowprops=dict(edgecolor="red", facecolor="red", arrowstyle="->", lw=1.5),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="red", facecolor="white"),
    )

    fig_jump_bin.tight_layout()
    fig_jump_bin.savefig("comparison_jump_vs_binary.pdf", bbox_inches="tight")
    fig_jump_bin.savefig("comparison_jump_vs_binary.png", bbox_inches="tight")
    figs.append(fig_jump_bin)

    # ------------------------------------------------------------------------
    # 4) BAR CHART #1: SINGLE FIXED SIZE
    #    -> WIDE AND SHORT
    # ------------------------------------------------------------------------
    fixed_n = 2**12  # e.g., 4096
    repetitions = 100
    linear_passes = []
    jump_passes = []
    binary_passes = []

    for _ in range(repetitions):
        arr = random.sample(range(0, fixed_n * 2), fixed_n)
        arr.sort()
        target = random.choice(arr)

        _, lin_steps = linear_search(arr, target)
        _, jum_steps = jump_search(arr, target)
        _, bin_steps = binary_search(arr, target)

        linear_passes.append(lin_steps)
        jump_passes.append(jum_steps)
        binary_passes.append(bin_steps)

    mean_linear = np.mean(linear_passes)
    mean_jump = np.mean(jump_passes)
    mean_binary = np.mean(binary_passes)

    fig_bar1 = plt.figure(figsize=(20, 4))  # wide & short
    algos = ["Linear Search", "Jump Search", "Binary Search"]
    values = [mean_linear, mean_jump, mean_binary]
    colors = ["orange", "green", "blue"]

    bars = plt.bar(algos, values, color=colors)
    plt.ylabel("Average Steps", fontsize=12)
    plt.title(f"Comparison at n = {fixed_n}", fontsize=14, fontweight="bold")

    for bar, val in zip(bars, values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig_bar1.tight_layout()
    fig_bar1.savefig("bar_fixed_n.pdf", bbox_inches="tight")
    fig_bar1.savefig("bar_fixed_n.png", bbox_inches="tight")
    figs.append(fig_bar1)

    # ------------------------------------------------------------------------
    # 5) BAR CHART #2: GROUPED BARS FOR MULTIPLE INPUT SIZES
    #    -> WIDE AND SHORT
    # ------------------------------------------------------------------------
    selected_sizes = [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
        16384,
        32768,
        65536,
    ]

    results_linear = []
    results_jump = []
    results_binary = []

    for size in selected_sizes:
        pass_lin = []
        pass_jum = []
        pass_bin = []
        for _ in range(15):
            arr = random.sample(range(0, size * 2), size)
            arr.sort()
            target = random.choice(arr)

            _, lin_steps = linear_search(arr, target)
            _, jum_steps = jump_search(arr, target)
            _, bin_steps = binary_search(arr, target)

            pass_lin.append(lin_steps)
            pass_jum.append(jum_steps)
            pass_bin.append(bin_steps)

        results_linear.append(np.mean(pass_lin))
        results_jump.append(np.mean(pass_jum))
        results_binary.append(np.mean(pass_bin))

    fig_bar2 = plt.figure(figsize=(20, 4))  # wide & short
    x = np.arange(len(selected_sizes))
    width = 0.25

    rects1 = plt.bar(
        x - width, results_linear, width, label="Linear Search", color="orange"
    )
    rects2 = plt.bar(x, results_jump, width, label="Jump Search", color="green")
    rects3 = plt.bar(
        x + width, results_binary, width, label="Binary Search", color="blue"
    )

    plt.xticks(x, selected_sizes, fontsize=8, rotation=45)
    plt.xlabel("Input Size (n)", fontsize=10)
    plt.ylabel("Average Steps", fontsize=10)
    plt.title(
        "Grouped Bar Chart: Average Steps for Many n", fontsize=12, fontweight="bold"
    )
    plt.legend(fontsize=9)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.7)

    def autolabel(rects):
        """Attach a text label above each bar, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.1,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig_bar2.tight_layout()
    fig_bar2.savefig("grouped_bar_many_sizes.pdf", bbox_inches="tight")
    fig_bar2.savefig("grouped_bar_many_sizes.png", bbox_inches="tight")
    figs.append(fig_bar2)

    return figs


def main():
    figures = generate_all_plots_and_save()

    pdf_filename = "Search_Comparison_Report.pdf"
    with PdfPages(pdf_filename) as pdf:
        for fig in figures:
            pdf.savefig(fig)
    print(f"Created consolidated PDF: {pdf_filename}")
    print("All figures saved as both PDF and PNG.")


if __name__ == "__main__":
    main()

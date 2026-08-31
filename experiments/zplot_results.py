#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R, P = ROOT / "results", ROOT / "plots"
P.mkdir(exist_ok=True)
read = lambda f: pd.read_csv(R / f)

TG = read("THREAD_task_granularity.csv")
RI = read("THREAD_regular_vs_irregular.csv")
TS = read("THREAD_scaling.csv").sort_values("threads")
OS = read("OPENMP_strong_scaling.csv").sort_values("threads")
OG = read("OPENMP_task_granularity.csv")
TW = read("OPENMP_task_vs_worksharing.csv")
MG = read("MPI_task_granularity.csv")
MS = read("MPI_strong_scaling.csv").sort_values("nodes")
MW = read("MPI_weak_scaling.csv").sort_values("nodes")
MRT_RAW = read("MPI_rank_thread_sweep_raw.csv")
MRT = read("MPI_rank_thread_sweep.csv")
MHB = read("MPI_hybrid_balance_sweep.csv")
SMALL = read("OPENMP_vs_THREAD_small_matrix_scaling.csv")
MPI_PHASES = [
    ("spmv_med", "SpMV"),
    ("normalize_med", "Normalize"),
    ("rotate_med", "Rotate"),
    ("communication_med", "Communication"),
    ("reduction_med", "Reduction"),
    ("epoch_med", "Epoch")
]

def draw(name, title, xlabel, ylabel, curves, ticks=None, logx=False, ideal=None):
    plt.figure(figsize=(7, 4.5))
    for x, y, label, style in curves:
        plt.plot(x, y, marker="o", linestyle=style, label=label)
    if ideal:
        plt.plot(ideal[0], ideal[1], "--", label=ideal[2])
    if logx:
        plt.xscale("log", base=2)
    if ticks is not None:
        plt.xticks(ticks, [str(x) for x in ticks])
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(P / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def groups(df, group, x, y, label):
    return [(d[x], d[y], label(v), "-") for v in sorted(df[group].unique())
            for d in [df[df[group] == v].sort_values(x)]]

def mpi_hybrid_balance_breakdown(name, df):

    d = df

    labels = [
        f"{int(row.mpi_processes)}r × {int(row.threads_per_process)}t"
        for row in d.itertuples()
    ]

    x = range(len(d))
    bottom = pd.Series(0.0, index=d.index)

    plt.figure(figsize=(9, 5))

    for col, label in MPI_PHASES:

        values = d[col].to_numpy()

        bars = plt.bar(
            x,
            values,
            bottom=bottom,
            label=label
        )

        # Write values inside each colored block
        plt.bar_label(
            bars,
            labels=[
                f"{value:.2f}" if value >= 0.03 else ""
                for value in values
            ],
            label_type="center",
            fontsize=8
        )

        bottom = bottom + d[col]

    n = int(d["n"].iloc[0])
    nz = int(d["nz"].iloc[0])
    nodes = int(d["nodes"].iloc[0])

    plt.title(
        f"MPI+OpenMP hybrid balance phase breakdown - {nodes} nodes\n"
        f"N = {n:,}, nnz = {nz:,}"
    )

    plt.xlabel("MPI ranks × OpenMP threads per rank")
    plt.ylabel("Time [s]")

    plt.xticks(
        list(x),
        labels
    )

    plt.grid(axis="y", alpha=.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        P / f"{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

def weak_scaling_breakdown(name, title, df, label_threshold=0.45):

    d = df.sort_values("nodes").copy()

    # Aggregate all local computation phases
    d["computation_med"] = (
        d["spmv_med"]
        + d["normalize_med"]
        + d["rotate_med"]
    )

    x = range(len(d))
    bottom = [0.0] * len(d)

    plt.figure(figsize=(9, 5))

    components = [
        ("computation_med", "Computation"),
        ("communication_med", "Communication"),
        ("reduction_med", "Reduction"),
        ("epoch_med", "Epoch Transition"),
        ("distribution_med", "Distribution")
    ]

    for col, label in components:

        values = d[col].to_numpy()

        bars = plt.bar(
            x,
            values,
            bottom=bottom,
            label=label
        )
        plt.bar_label(
            bars,
            labels=[
                f"{value:.2f}" if value > label_threshold else ""
                for value in values
            ],
            label_type="center",
            fontsize=9,
            fontweight="bold"
        )

        bottom = [
            b + v
            for b, v in zip(bottom, values)
        ]

    plt.title(title)
    plt.xlabel("Number of Nodes")
    plt.ylabel("Time [s]")

    plt.xticks(
        list(x),
        d["nodes"].astype(str)
    )

    plt.grid(axis="y", alpha=.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        P / f"{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()
    
def plot_small_irregular_scaling(name, df):
    d = df[
        (df["mode"] == "irregular") &
        (df["threads"].isin([2, 4, 8]))
    ].sort_values("threads").copy()

    # Media dei 3 run
    d["thread_mean"] = d[
        ["thread_run1", "thread_run2", "thread_run3"]
    ].mean(axis=1)

    d["openmp_mean"] = d[
        ["openmp_run1", "openmp_run2", "openmp_run3"]
    ].mean(axis=1)

    # Deviazione standard per le error bar
    d["thread_std"] = d[
        ["thread_run1", "thread_run2", "thread_run3"]
    ].std(axis=1)

    d["openmp_std"] = d[
        ["openmp_run1", "openmp_run2", "openmp_run3"]
    ].std(axis=1)

    n = int(d["n"].iloc[0])
    nz = int(d["nz"].iloc[0])

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        d["threads"],
        d["thread_mean"],
        yerr=d["thread_std"],
        marker="o",
        capsize=4,
        label="C++ Threads"
    )

    plt.errorbar(
        d["threads"],
        d["openmp_mean"],
        yerr=d["openmp_std"],
        marker="o",
        capsize=4,
        label="OpenMP"
    )

    plt.xlabel("number of threads")
    plt.ylabel("execution time [s]")

    plt.title(
        f"Strong scaling - irregular workload - "
        f"N = {n:,}, nnz = {nz:,} - execution time"
    )

    plt.xticks([2, 4, 8])
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(P / f"{name}.png", dpi=300)
    plt.close()

# MPI rank/thread phase breakdown
def mpi_rank_thread_breakdown(name, mode, df):

    d = (
        df[df["mode"] == mode]
        .copy()
        .sort_values(["mpi_processes", "threads_per_process"])
    )

    labels = [
        f"{int(row.mpi_processes)}r × {int(row.threads_per_process)}t"
        for row in d.itertuples()
    ]

    x = range(len(d))
    bottom = pd.Series(0.0, index=d.index)

    plt.figure(figsize=(9, 5))

    for col, label in MPI_PHASES:
        plt.bar(
            x,
            d[col],
            bottom=bottom,
            label=label
        )

        bottom = bottom + d[col]

    n = int(d["n"].iloc[0])
    nz = int(d["nz"].iloc[0])

    plt.title(
        f"MPI+OpenMP phase breakdown - {mode} workload\n"
        f"N = {n:,}, nnz = {nz:,}"
    )

    plt.xlabel("MPI ranks × OpenMP threads per rank")
    plt.ylabel("Time [s]")

    plt.xticks(
        list(x),
        labels
    )

    plt.grid(axis="y", alpha=.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        P / f"{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

# MPI rank/thread sweep
def mpi_rank_thread_sweep(name, mode, raw):

    df = raw[raw["mode"] == mode].copy()

    plt.figure(figsize=(7, 4.5))

    for ranks in sorted(df["mpi_processes"].unique()):

        d = df[df["mpi_processes"] == ranks]

        stats = (
            d.groupby("threads_per_process")["time"]
             .agg(["mean", "min", "max"])
             .reset_index()
             .sort_values("threads_per_process")
        )

        plt.errorbar(
            stats["threads_per_process"],
            stats["mean"],
            yerr=[
                stats["mean"] - stats["min"],
                stats["max"] - stats["mean"]
            ],
            marker="o",
            capsize=4,
            linewidth=2,
            label=f"{ranks} MPI rank" if ranks == 1 else f"{ranks} MPI ranks"
        )

    n = int(df["n"].iloc[0])
    nz = int(df["nz"].iloc[0])

    threads = sorted(df["threads_per_process"].unique())

    plt.title(
        f"MPI+OpenMP rank/thread interaction - {mode} workload\n"
        f"N = {n:,}, nnz = {nz:,}"
    )

    plt.xlabel("OpenMP threads per rank")
    plt.ylabel("Execution time [s]")

    plt.xticks(threads)
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        P / f"{name}.png",
        dpi=220,
        bbox_inches="tight"
    )

    plt.close()

b = sorted(TG.block_size.unique())
draw("01_thread_granularity", "C++ Threads granularity - irregular", "Block size", "Time [s]",
     groups(TG, "threads", "block_size", "time_med", lambda t: f"{t} threads"), b, True)

draw("02_regular_vs_irregular", "C++ Threads: regular vs irregular", "Threads", "Time [s]",
     [(d.threads, d.thread_time_med, m.capitalize(), "-") for m in ("regular", "irregular")
      for d in [RI[RI["mode"] == m].sort_values("threads")]], sorted(RI.threads.unique()))

draw(
    "03_threads_vs_openmp_time",
    "C++ Threads vs OpenMP task-based",
    "Threads",
    "Time [s]",
    [
        (TS.threads, TS.thread_time_med, "C++ Threads", "-"),
        (OS.threads, OS.openmp_time_med, "OpenMP", "--")
    ],
    list(TS.threads)
)

draw("04_threads_vs_openmp_speedup", "C++ Threads vs OpenMP speedup", "Threads", "Speedup",
     [(OS.threads, OS.thread_speedup, "C++ Threads", "-"),
      (OS.threads, OS.openmp_speedup, "OpenMP", "-")], list(OS.threads),
     ideal=(OS.threads, OS.threads, "Ideal"))

b = sorted(OG.block_size.unique())
draw("05_openmp_granularity", "OpenMP task granularity - irregular", "Block size", "Time [s]",
     groups(OG, "threads", "block_size", "openmp_time_med", lambda t: f"{t} threads"), b, True)

curves = []
for t in sorted(TW.threads.unique()):
    d = TW[TW.threads == t].sort_values("block_size")
    curves += [(d.block_size, d.task_time_med, f"Task, {t}T", "-"),
               (d.block_size, d.worksharing_time_med, f"Work-sharing, {t}T", "--")]
draw("06_openmp_task_vs_worksharing", "OpenMP task-based vs work-sharing", "Block size", "Time [s]",
     curves, sorted(TW.block_size.unique()), True)

# MPI + OpenMP
b = sorted(MG.block_size.unique())
draw("07_mpi_granularity", "MPI+OpenMP granularity - 8 nodes", "Block size", "Time [s]",
     groups(MG, "threads_per_process", "block_size", "time_med", lambda t: f"{t} threads/rank"), b, True)

n = list(MS.nodes)
draw("08_mpi_strong", "MPI+OpenMP strong scaling", "Nodes", "Time [s]",
     [(MS.nodes, MS.mpi_time_med, "Measured", "-")], n,
     ideal=(MS.nodes, MS.mpi_time_med.iloc[0] / MS.nodes, "Ideal T(1)/p"))

draw("09_mpi_speedup", "MPI+OpenMP relative speedup", "Nodes", "Speedup",
     [(MS.nodes, MS.relative_speedup, "Measured", "-")], n,
     ideal=(MS.nodes, MS.nodes, "Ideal"))

n = list(MW.nodes)
draw("10_mpi_weak", "MPI+OpenMP weak scaling", "Nodes", "Time [s]",
     [(MW.nodes, MW.time_med, "Measured", "-")], n,
     ideal=(MW.nodes, [MW.time_med.iloc[0]] * len(MW), "Ideal constant time"))

weak_scaling_breakdown(
    "17_mpi_weak_scaling_breakdown",
    "MPI+OpenMP Weak Scalability Time Breakdown",
    MW, label_threshold=0.45
)

plot_small_irregular_scaling(
    "threads_vs_openmp_small_irregular",
    SMALL
)

mpi_rank_thread_breakdown(
    "11_mpi_rank_thread_breakdown_regular",
    "regular",
    MRT
)

mpi_rank_thread_breakdown(
    "12_mpi_rank_thread_breakdown_irregular",
    "irregular",
    MRT
)


mpi_rank_thread_sweep(
    "13_mpi_rank_thread_sweep_regular",
    "regular",
    MRT_RAW
)

mpi_rank_thread_sweep(
    "14_mpi_rank_thread_sweep_irregular",
    "irregular",
    MRT_RAW
)
# MPI regular vs irregular comparison
curves = []

for threads in sorted(MRT["threads_per_process"].unique()):

    for mode, style in [
        ("regular", "-"),
        ("irregular", "--")
    ]:

        d = (
            MRT[
                (MRT["threads_per_process"] == threads) &
                (MRT["mode"] == mode)
            ]
            .sort_values("mpi_processes")
        )

        curves.append(
            (
                d["mpi_processes"],
                d["time_med"],
                f"{mode.capitalize()}, {threads} threads/rank",
                style
            )
        )

draw(
    "15_mpi_regular_vs_irregular",
    "MPI+OpenMP: regular vs irregular workload",
    "MPI processes",
    "Time [s]",
    curves,
    sorted(MRT["mpi_processes"].unique())
)
mpi_hybrid_balance_breakdown(
    "16_mpi_hybrid_balance_breakdown",
    MHB
)

print(f"Plots saved in: {P}")
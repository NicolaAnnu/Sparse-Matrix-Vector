#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
R, P = ROOT / "results", ROOT / "plots"
P.mkdir(exist_ok=True)
read = lambda f: pd.read_csv(R / f)


def draw(name, title, xlabel, ylabel, curves, ticks=None, logx=False, ideal=None, note=None):
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
    if note:
        plt.figtext(.5, -.02, note, ha="center", fontsize=8)
    plt.savefig(P / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


def groups(df, group, x, y, label):
    return [(d[x], d[y], label(v), "-") for v in sorted(df[group].unique())
            for d in [df[df[group] == v].sort_values(x)]]


TG = read("THREAD_task_granularity.csv")
RI = read("THREAD_regular_vs_irregular.csv")
TS = read("THREAD_scaling.csv").sort_values("threads")
OS = read("OPENMP_scaling.csv").sort_values("threads")
OG = read("OPENMP_task_granularity.csv")
TW = read("OPENMP_task_vs_worksharing.csv")
MG = read("MPI_task_granularity.csv")
MS = read("MPI_strong_scaling.csv").sort_values("nodes")
MW = read("MPI_weak_scaling.csv").sort_values("nodes")

# C++ Threads / OpenMP
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

def breakdown(name, title, df):
    x = df["nodes"]
    bottom = 0
    plt.figure(figsize=(8, 4.8))

    for col, label in [
        ("distribution_med", "Distribution"),
        ("local_computation_med", "Local computation"),
        ("communication_med", "Communication"),
        ("reduction_med", "Reduction"),
        ("epoch_med", "Epoch")
    ]:
        plt.bar(x, df[col], bottom=bottom, label=label)
        bottom = bottom + df[col]

    plt.title(title)
    plt.xlabel("Number of MPI processes / nodes")
    plt.ylabel("Time [s]")
    plt.xticks(x)
    plt.grid(axis="y", alpha=.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(P / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close()


breakdown(
    "11_mpi_strong_breakdown",
    "MPI+OpenMP phase breakdown - strong scaling - irregular workload",
    MS
)

breakdown(
    "12_mpi_weak_breakdown",
    "MPI+OpenMP phase breakdown - weak scaling - irregular workload",
    MW
)

print(f"Plots saved in: {P}")
"""
GPU Scheduler Simulator
=======================
Compares FIFO vs Priority vs Work-Stealing scheduling strategies
for AI/ML workloads across multi-GPU systems.

Metrics tracked:
  - Makespan (total wall-clock time to finish all jobs)
  - GPU idle % (wasted compute capacity)
  - Average wait time for high-priority jobs
  - Per-GPU utilization breakdown
  - GPU 0 utilization timeline

Run:
  python gpu_scheduler_sim.py
"""

import random
import simpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ── config ────────────────────────────────────────────────────────────────────

random.seed(42)
np.random.seed(42)

NUM_GPUS       = 4
NUM_JOBS       = 30
HIGH_PRI_PCT   = 0.30    # 30 % of jobs are high-priority
DURATION_MIN   = 2
DURATION_MAX   = 12
ARRIVAL_SPREAD = 15      # jobs arrive within this time window

COLORS = {
    "FIFO":          "#378ADD",
    "Priority":      "#9F4DCA",
    "Work-Stealing": "#1D9E75",
}
HI_PRI_COLOR = "#E24B4A"
LO_PRI_COLOR = "#888780"

# ── job generation ─────────────────────────────────────────────────────────────

def generate_jobs(n=NUM_JOBS, hi_pct=HIGH_PRI_PCT, seed=42):
    random.seed(seed)
    jobs = []
    for i in range(n):
        duration    = random.randint(DURATION_MIN, DURATION_MAX)
        priority    = 1 if random.random() < hi_pct else 0   # 1 = high
        arrival     = random.uniform(0, ARRIVAL_SPREAD)
        jobs.append({
            "id":       i,
            "duration": duration,
            "priority": priority,
            "arrival":  round(arrival, 2),
        })
    return jobs


# ── simulation engines ─────────────────────────────────────────────────────────

class SchedulerBase:
    """Shared infrastructure: GPU resources, event log, metric collection."""

    def __init__(self, jobs, num_gpus=NUM_GPUS):
        self.jobs       = jobs
        self.num_gpus   = num_gpus
        self.env        = simpy.Environment()
        self.gpus       = [simpy.Resource(self.env, capacity=1) for _ in range(num_gpus)]
        self.results    = []                        # completed job records
        self.gpu_busy   = defaultdict(float)        # total busy time per GPU
        self.timeline   = defaultdict(list)         # (start, end, priority) per GPU

    def record(self, job, gpu_id, start, end):
        wait = start - job["arrival"]
        self.results.append({**job, "gpu": gpu_id, "start": start,
                              "end": end, "wait": max(0, wait)})
        self.gpu_busy[gpu_id] += job["duration"]
        self.timeline[gpu_id].append((start, end, job["priority"]))

    def run(self):
        raise NotImplementedError

    def metrics(self):
        if not self.results:
            return {}
        makespan     = max(r["end"] for r in self.results)
        total_cap    = makespan * self.num_gpus
        total_busy   = sum(self.gpu_busy.values())
        idle_pct     = (1 - total_busy / total_cap) * 100 if total_cap else 0
        hi            = [r for r in self.results if r["priority"] == 1]
        avg_wait_hi  = np.mean([r["wait"] for r in hi]) if hi else 0
        avg_wait_all = np.mean([r["wait"] for r in self.results])
        util_per_gpu = {g: self.gpu_busy[g] / makespan * 100 for g in range(self.num_gpus)}
        return {
            "makespan":      round(makespan, 2),
            "idle_pct":      round(idle_pct, 1),
            "avg_wait_hi":   round(avg_wait_hi, 2),
            "avg_wait_all":  round(avg_wait_all, 2),
            "util_per_gpu":  util_per_gpu,
        }


class FIFOScheduler(SchedulerBase):
    """Jobs dispatched strictly in arrival order; first free GPU wins."""

    def run(self):
        queue = sorted(self.jobs, key=lambda j: j["arrival"])

        def dispatch(job):
            yield self.env.timeout(job["arrival"])
            gpu_id = min(range(self.num_gpus),
                         key=lambda g: self.gpus[g].count + len(self.gpus[g].queue))
            with self.gpus[gpu_id].request() as req:
                yield req
                start = self.env.now
                yield self.env.timeout(job["duration"])
                self.record(job, gpu_id, start, self.env.now)

        for job in queue:
            self.env.process(dispatch(job))
        self.env.run()


class PriorityScheduler(SchedulerBase):
    """High-priority jobs jump ahead of low-priority jobs in the queue."""

    def run(self):
        # simpy.PriorityResource: lower number = higher priority → invert our flag
        self.gpus = [simpy.PriorityResource(self.env, capacity=1)
                     for _ in range(self.num_gpus)]

        def dispatch(job):
            yield self.env.timeout(job["arrival"])
            prio_val = 0 if job["priority"] == 1 else 1   # high-pri → 0 (runs first)
            gpu_id = min(range(self.num_gpus),
                         key=lambda g: self.gpus[g].count + len(self.gpus[g].queue))
            with self.gpus[gpu_id].request(priority=prio_val) as req:
                yield req
                start = self.env.now
                yield self.env.timeout(job["duration"])
                self.record(job, gpu_id, start, self.env.now)

        for job in sorted(self.jobs, key=lambda j: j["arrival"]):
            self.env.process(dispatch(job))
        self.env.run()


class WorkStealingScheduler(SchedulerBase):
    """
    Each GPU owns a local FIFO queue.
    Jobs initially assigned round-robin.
    An idle GPU steals from the longest queue.
    """

    def run(self):
        sorted_jobs = sorted(self.jobs, key=lambda j: j["arrival"])
        local_queues = [[] for _ in range(self.num_gpus)]

        # Round-robin initial assignment
        for i, job in enumerate(sorted_jobs):
            local_queues[i % self.num_gpus].append(job)

        def worker(gpu_id):
            while True:
                # Try own queue first
                if local_queues[gpu_id]:
                    job = local_queues[gpu_id].pop(0)
                else:
                    # Steal from busiest queue
                    victim = max(range(self.num_gpus),
                                 key=lambda g: len(local_queues[g]))
                    if not local_queues[victim]:
                        break                         # all queues empty → done
                    job = local_queues[victim].pop()  # steal from tail

                # Wait until job has arrived
                if self.env.now < job["arrival"]:
                    yield self.env.timeout(job["arrival"] - self.env.now)

                with self.gpus[gpu_id].request() as req:
                    yield req
                    start = self.env.now
                    yield self.env.timeout(job["duration"])
                    self.record(job, gpu_id, start, self.env.now)

        for g in range(self.num_gpus):
            self.env.process(worker(g))
        self.env.run()


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_results(schedulers: dict):
    names    = list(schedulers.keys())
    metrics  = {n: schedulers[n].metrics() for n in names}
    clrs     = [COLORS[n] for n in names]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#f8f8f6")

    # ── Title ──────────────────────────────────────────────────────────────────
    fig.suptitle(
        "GPU Scheduler Simulator  ·  FIFO vs Priority vs Work-Stealing",
        fontsize=16, fontweight="bold", y=0.98, color="#2C2C2A"
    )
    fig.text(
        0.5, 0.955,
        f"{NUM_GPUS} GPUs  ·  {NUM_JOBS} jobs  ·  {int(HIGH_PRI_PCT*100)}% high-priority  "
        f"·  job duration {DURATION_MIN}–{DURATION_MAX} units",
        ha="center", fontsize=11, color="#5F5E5A"
    )

    gs = fig.add_gridspec(4, 3, hspace=0.55, wspace=0.35,
                          top=0.92, bottom=0.06, left=0.07, right=0.97)

    bar_kw = dict(color=clrs, edgecolor="none", zorder=3)

    # ── 1. Makespan ────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [metrics[n]["makespan"] for n in names]
    bars = ax1.bar(names, vals, **bar_kw)
    best = min(vals)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                 color="#3B6D11" if v == best else "#444441", fontweight="bold" if v==best else "normal")
    ax1.set_title("Makespan (lower = faster)", fontsize=10, pad=6)
    ax1.set_ylabel("time units", fontsize=9)
    ax1.set_ylim(0, max(vals) * 1.25)
    ax1.set_facecolor("#f0f0ee")
    ax1.grid(axis="y", color="white", linewidth=0.8)
    ax1.tick_params(labelsize=9)

    # ── 2. GPU idle % ──────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [metrics[n]["idle_pct"] for n in names]
    bars = ax2.bar(names, vals, **bar_kw)
    best = min(vals)
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=9,
                 color="#3B6D11" if v == best else "#444441", fontweight="bold" if v==best else "normal")
    ax2.set_title("GPU idle % (lower = better)", fontsize=10, pad=6)
    ax2.set_ylabel("idle %", fontsize=9)
    ax2.set_ylim(0, max(vals) * 1.35)
    ax2.set_facecolor("#f0f0ee")
    ax2.grid(axis="y", color="white", linewidth=0.8)
    ax2.tick_params(labelsize=9)

    # ── 3. Avg wait — high priority ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    vals = [metrics[n]["avg_wait_hi"] for n in names]
    bars = ax3.bar(names, vals, **bar_kw)
    best = min(vals)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, v + 0.05,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=9,
                 color="#3B6D11" if v == best else "#444441", fontweight="bold" if v==best else "normal")
    ax3.set_title("Avg wait — high-priority jobs\n(lower = better)", fontsize=10, pad=6)
    ax3.set_ylabel("time units", fontsize=9)
    ax3.set_ylim(0, max(vals) * 1.35 + 0.5)
    ax3.set_facecolor("#f0f0ee")
    ax3.grid(axis="y", color="white", linewidth=0.8)
    ax3.tick_params(labelsize=9)

    # ── 4. Per-GPU utilisation ─────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    gpu_ids = list(range(NUM_GPUS))
    x = np.arange(NUM_GPUS)
    width = 0.25
    for i, name in enumerate(names):
        util = [metrics[name]["util_per_gpu"].get(g, 0) for g in gpu_ids]
        ax4.bar(x + (i - 1) * width, util, width, label=name,
                color=COLORS[name], edgecolor="none", zorder=3)
    ax4.set_title("Per-GPU utilisation (higher = better)", fontsize=10, pad=6)
    ax4.set_ylabel("utilisation %", fontsize=9)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"GPU {g}" for g in gpu_ids], fontsize=9)
    ax4.set_ylim(0, 120)
    ax4.legend(fontsize=9, framealpha=0)
    ax4.set_facecolor("#f0f0ee")
    ax4.grid(axis="y", color="white", linewidth=0.8)
    ax4.tick_params(labelsize=9)

    # ── 5. Timeline: GPU 0 per scheduler ──────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :])
    max_end = max(
        max((seg[1] for seg in schedulers[n].timeline[0]), default=0)
        for n in names
    )
    row_h, gap = 0.6, 0.15
    yticks, ylabels = [], []

    for i, name in enumerate(names):
        y = i * (row_h + gap)
        yticks.append(y + row_h / 2)
        ylabels.append(name)
        segs = schedulers[name].timeline[0]
        for (start, end, prio) in segs:
            fc = HI_PRI_COLOR if prio == 1 else COLORS[name]
            rect = mpatches.FancyBboxPatch(
                (start, y), end - start, row_h,
                boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor="white", linewidth=0.5, zorder=3
            )
            ax5.add_patch(rect)

    ax5.set_xlim(0, max_end * 1.02)
    ax5.set_ylim(-gap, len(names) * (row_h + gap))
    ax5.set_yticks(yticks)
    ax5.set_yticklabels(ylabels, fontsize=10)
    ax5.set_xlabel("time units", fontsize=9)
    ax5.set_title("GPU 0 utilisation timeline  (red = high-priority job)", fontsize=10, pad=6)
    ax5.set_facecolor("#f0f0ee")
    ax5.grid(axis="x", color="white", linewidth=0.8)
    ax5.tick_params(axis="y", left=False, labelsize=9)

    # legend for timeline
    patches = [
        mpatches.Patch(color=HI_PRI_COLOR, label="High-priority job"),
        mpatches.Patch(color="#888780",    label="Normal job"),
    ]
    ax5.legend(handles=patches, fontsize=9, framealpha=0, loc="upper right")

    # ── 6. Insight summary table ───────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis("off")

    col_labels = ["Scheduler", "Makespan", "GPU idle %", "Hi-pri wait", "Best for"]
    best_desc  = {
        "FIFO":          "predictability & simplicity",
        "Priority":      "latency-sensitive inference workloads",
        "Work-Stealing": "throughput-heavy training workloads",
    }
    rows = []
    for name in names:
        m = metrics[name]
        rows.append([
            name,
            f"{m['makespan']:.1f} units",
            f"{m['idle_pct']:.1f}%",
            f"{m['avg_wait_hi']:.2f} units",
            best_desc[name],
        ])

    tbl = ax6.table(
        cellText=rows, colLabels=col_labels,
        cellLoc="center", loc="center",
        bbox=[0, 0.1, 1, 0.85]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D3D1C7")
        if r == 0:
            cell.set_facecolor("#2C2C2A")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0ee")
        else:
            cell.set_facecolor("#fafaf8")
        if c == 0 and r > 0:
            cell.set_text_props(color=list(COLORS.values())[r - 1], fontweight="bold")

    ax6.set_title("Summary — when to use each strategy", fontsize=10, pad=6)

    plt.savefig("/home/claude/gpu_scheduler_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Saved → gpu_scheduler_results.png")
    plt.show()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Generating {NUM_JOBS} jobs ({int(HIGH_PRI_PCT*100)}% high-priority) "
          f"across {NUM_GPUS} GPUs...\n")

    jobs = generate_jobs()

    schedulers = {
        "FIFO":          FIFOScheduler(jobs, NUM_GPUS),
        "Priority":      PriorityScheduler(jobs, NUM_GPUS),
        "Work-Stealing": WorkStealingScheduler(jobs, NUM_GPUS),
    }

    for name, sched in schedulers.items():
        sched.run()
        m = sched.metrics()
        print(f"[{name}]")
        print(f"  Makespan:          {m['makespan']} units")
        print(f"  GPU idle %:        {m['idle_pct']}%")
        print(f"  Avg wait (hi-pri): {m['avg_wait_hi']} units")
        print(f"  Avg wait (all):    {m['avg_wait_all']} units")
        print()

    plot_results(schedulers)


if __name__ == "__main__":
    main()

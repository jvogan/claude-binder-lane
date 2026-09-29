"""Printed progress narration for a sharded science campaign.

Claude Science has no progress call, and polling is forbidden. The campaign
narrates itself by printing. This module owns that narration. It prints one
fixed width line per event, labels every estimate, stays silent while
containers work, and closes with a summary block.

Standard library only. Top level is definitions plus width constants, so
exec() runs this body straight into a kernel namespace. That is weaker than
the kernel.py sidecar contract: the documented AST gate for sidecars takes
definitions and literal constants, and the derived widths below are
arithmetic, so this file rides as an imported module or an explicit exec,
not as an injected kernel.py.

The two timing constants below trace to one measured run. Any figure built
from them is arithmetic, not measurement, so every such figure prints with
an est label.
"""

import math
import sys
import time
from statistics import median

# Measured once on the fal H100 canary. One sample each, not a distribution,
# which is why every figure derived from them carries an est label.
LOAD_S = 142.849  # model load, paid once per cold container
FOLD_S = 35.744   # seconds per fold at 190 total residues


# --------------------------------------------------------------------------
# Grid. Every line is: elapsed | slot | scope | message.
# The slot holds an event tag or a row label, so sublines align under tags.
# --------------------------------------------------------------------------

W_CLOCK = 9
W_SLOT = 9
W_SCOPE = 24
_MSG_PAD = " " * (W_SCOPE + 2)
_CONT = " " * W_CLOCK + " "


def fmt_clock(seconds):
    """Elapsed wall clock since run start, measured by this process."""
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return "t+%d:%02d:%02d" % (h, m, s)
    return "t+%02d:%02d" % (m, s)


def fmt_secs(x, est=True):
    """Seconds, carrying an est label unless the caller passes est=False.

    Estimates render coarse on purpose. Measurements keep full precision,
    because they are the numbers of record.
    """
    if est:
        return "%.1f s est" % x
    return "%.3f s" % x


def _min_val(x):
    if x < 600:
        return "%.1f" % (x / 60.0)
    return "%d" % int(round(x / 60.0))


def fmt_min(x, est=True):
    """Coarse minutes, labeled est unless the caller passes est=False."""
    return "%s min%s" % (_min_val(x), " est" if est else "")


def fmt_min_range(lo, hi, est=True):
    """A minute range with one est label covering both ends.

    Collapses to a single figure when the ends agree, so a queue of one
    short shard never prints the same number twice.
    """
    if abs(hi - lo) < 0.05:
        return fmt_min(lo, est=est)
    label = "est " if est else ""
    return "%s%s to %s min" % (label, _min_val(lo), _min_val(hi))


def shard_wall(k_folds):
    """Estimated wall seconds for one shard of k folds. Arithmetic on L and F."""
    return LOAD_S + k_folds * FOLD_S


def run_plan(n_designs, seeds, shard_size):
    """Return (total_folds, n_shards, gpu_seconds_estimate)."""
    total = n_designs * seeds
    shards = int(math.ceil(total / float(shard_size))) if total else 0
    gpu = shards * LOAD_S + total * FOLD_S
    return total, shards, gpu


def largest_fit(rate_usd_per_gpu_s, remaining_usd, seeds, shard_size, n_max):
    """Largest design count whose estimated cost fits the remaining budget.

    Returns None when even one design does not fit. Pure scan, no guessing.
    """
    for n in range(n_max, 0, -1):
        _, _, gpu = run_plan(n, seeds, shard_size)
        if gpu * rate_usd_per_gpu_s <= remaining_usd:
            return n
    return None


class Progress(object):
    """Collects one run's events and prints them on a fixed grid.

    A clock callable can be injected for tests and demos. By default the
    clock is monotonic time, and elapsed figures are genuine measurements.
    """

    def __init__(self, out=None, clock=None):
        self.out = out if out is not None else sys.stdout
        self._clock = clock if clock is not None else time.monotonic
        self.t0 = self._clock()
        self.run_id = ""
        self.n = 0
        self.s = 0
        self.k = 0
        self.tier = ""
        self.cap = 1
        self.total = 0
        self.shards = 0
        self.gpu_est = 0.0
        # Rate parameters arrive from config at run time. None means no
        # verified rate exists, and then no dollar figure is ever printed.
        self.rate = None
        self.ceiling = None
        self.spent = None
        self.spent_kind = "estimated"

    # ---------------- infrastructure ----------------

    def _elapsed(self):
        return self._clock() - self.t0

    def _emit(self, text):
        self.out.write(text.rstrip() + "\n")

    def _line(self, tag, scope, message):
        stamp = fmt_clock(self._elapsed())
        self._emit("%-*s %-*s %-*s  %s" % (
            W_CLOCK, stamp, W_SLOT, tag, W_SCOPE, scope, message))

    def _row(self, label, message):
        self._emit("%s%-*s %s%s" % (_CONT, W_SLOT, label, _MSG_PAD, message))

    # ---------------- events ----------------

    def configure(self, run_id, n_designs, seeds, shard_size, tier,
                  concurrency_cap, rate=None, ceiling=None, spent=None,
                  spent_kind="estimated"):
        """Record the plan. No printing. Call accept() next."""
        self.run_id = run_id
        self.n = n_designs
        self.s = seeds
        self.k = shard_size
        self.tier = tier
        self.cap = max(1, concurrency_cap)
        self.rate = rate
        self.ceiling = ceiling
        self.spent = spent
        self.spent_kind = spent_kind
        self.total, self.shards, self.gpu_est = run_plan(n_designs, seeds,
                                                         shard_size)

    def gate_verdict(self):
        """Return "pass", "fail" or "no_rate" for the standing budget gate.

        The validator rejects only estimated greater than maximum, equality
        passes. NEXT8.md:63.
        """
        if self.rate is None or self.ceiling is None:
            return "no_rate"
        remaining = self.ceiling - (self.spent or 0.0)
        if self.gpu_est * self.rate > remaining:
            return "fail"
        return "pass"

    def accept(self):
        """Event: run accepted. States the plan and the timing provenance."""
        self._line("ACCEPT", self.run_id,
                   "%d designs x %d seeds = %d folds, tier %s"
                   % (self.n, self.s, self.total, self.tier))
        self._row("plan", "%d shards of %d folds, cap %d handles"
                  % (self.shards, self.k, self.cap))
        self._row("",
                  "gpu %s, wall %s"
                  % (fmt_secs(self.gpu_est), self._wall_range_text()))
        self._row("timing",
                  "load %s and fold %s, each measured once;"
                  " the measurement record does not ship"
                  % (fmt_secs(LOAD_S, est=False),
                     fmt_secs(FOLD_S, est=False)))
        self._row("",
                  "every time figure below is arithmetic on those two numbers")
        verdict = self.gate_verdict()
        if verdict == "pass":
            usd = self.gpu_est * self.rate
            remaining = self.ceiling - (self.spent or 0.0)
            self._row("gate",
                      "est %.2f USD fits %.2f USD remaining after %s spend"
                      " [PARAMETER, unverified]"
                      % (usd, remaining, self.spent_kind))
        elif verdict == "no_rate":
            self._row("gate",
                      "dollars unavailable this run, no verified rate"
                      " parameter")
        self._silly_shard_note()

    def _wall_range_text(self):
        """Honest wall range. Lower bound assumes perfect packing at the cap.

        Upper bound assumes every handle waits on one longest shard. Both are
        arithmetic on single-observation timings, hence est.
        """
        walls = self._shard_fold_counts()
        longest = shard_wall(max(walls)) if walls else 0.0
        lo = self.gpu_est / float(self.cap)
        waves = int(math.ceil(self.shards / float(self.cap))) if self.shards \
            else 0
        hi = waves * longest
        return fmt_min_range(lo, hi)

    def _shard_fold_counts(self):
        counts = []
        left = self.total
        while left > 0:
            take = min(self.k, left)
            counts.append(take)
            left -= take
        return counts

    def _silly_shard_note(self):
        """Warn when a trailing shard pays a full load for few folds."""
        counts = self._shard_fold_counts()
        if len(counts) < 2:
            return
        tail = counts[-1]
        share = LOAD_S / shard_wall(tail)
        if share >= 0.5:
            self._row("note",
                      "trailing shard holds %d %s and is %.0f percent load;"
                      " consider %d or fewer designs, or a fuller batch"
                      % (tail, "fold" if tail == 1 else "folds",
                         share * 100.0,
                         ((len(counts) - 1) * self.k) // self.s))

    def submit(self, shard_index, job_id, fold_count):
        """Event: shard submitted. After this line the module goes quiet."""
        noun = "fold" if fold_count == 1 else "folds"
        self._line("SUBMIT", "shard %d/%d" % (shard_index, self.shards),
                   "job %s, %d %s, wall %s"
                   % (job_id, fold_count, noun,
                      fmt_min(shard_wall(fold_count))))

    def landed(self, shard_index, job_id, fold_count, gpu_measured=None,
               harvest=None):
        """Event: shard landed, driven by the host completion notification.

        gpu_measured is the job's own reported GPU seconds. When a job does
        not report timings, the line says so instead of substituting an est.
        """
        if gpu_measured is None:
            gpu_text = "gpu not reported by job"
        else:
            gpu_text = "gpu %s measured" % fmt_secs(gpu_measured, est=False)
        extra = ", harvest %s" % harvest if harvest else ""
        self._line("LANDED", "shard %d/%d" % (shard_index, self.shards),
                   "job %s, %d folds, %s%s"
                   % (job_id, fold_count, gpu_text, extra))
        self._remain_row(shard_index)

    def _remain_row(self, landed_count):
        """Recompute remaining time from what is still pending. Never precise.

        The range spans perfect packing at the cap down to one handle serial.
        It shrinks only when notifications arrive, because nothing else is
        observable without polling.
        """
        pending = self.shards - landed_count
        if pending <= 0:
            return
        counts = self._shard_fold_counts()[landed_count:]
        walls = [shard_wall(c) for c in counts]
        total_pending = sum(walls)
        hands = min(self.cap, pending)
        lo = total_pending / float(hands)
        hi = total_pending  # one handle doing everything, the safe ceiling
        noun = "shard" if pending == 1 else "shards"
        self._line("REMAIN", "",
                   "%d %s pending, %s" % (pending, noun,
                                          fmt_min_range(lo, hi)))

    def scored(self, design, results):
        """Event: design scored. One line per design, never one per fold.

        results is a list of dicts with seed, iptm and plddt. Seeds are
        aggregated here so a 5 seed design costs one line, not five.
        """
        iptms = sorted(r["iptm"] for r in results)
        plddts = sorted(r["plddt"] for r in results)
        self._line("SCORED", design,
                   "%d seeds, iptm med %.3f rng %.3f-%.3f, plddt med %.3f"
                   % (len(results), median(iptms), iptms[0], iptms[-1],
                      median(plddts)))

    def failed(self, shard_index, job_id, stage, detail, lost, kept, actions):
        """Event: a shard or seed failed.

        Names what failed, which designs and seeds are lost, what survived,
        and what the user can do. Silence and bare stack traces are both wrong.
        """
        self._line("FAILED", "shard %d/%d" % (shard_index, self.shards),
                   "job %s, %s: %s" % (job_id, stage, detail))
        self._row("what", "%s" % lost)
        self._row("kept", "%s" % kept)
        for i, act in enumerate(actions):
            self._row("action" if i == 0 else "", act)

    def gate_failed(self, requested_n=None):
        """Event: budget gate refused the run before any submit.

        Mirrors the refusal shape in the cost-estimator design note, which
        does not ship. Kind tags tell estimate from measurement.
        Called without a rate or ceiling, it says the gate was never checked
        instead of crashing on dollar arithmetic that has no inputs.
        """
        gated = self.rate is not None and self.ceiling is not None
        n_show = requested_n if requested_n is not None else self.n
        if gated:
            headline = "estimate exceeds remaining ceiling, nothing submitted"
            remaining = self.ceiling - (self.spent or 0.0)
            usd = self.gpu_est * self.rate
        else:
            headline = ("no rate parameter, so the ceiling was never "
                        "checked; nothing submitted")
        self._line("GATEFAIL", self.run_id, headline)
        self._row("tier", "%s [PARAMETER]" % self.tier)
        self._row("asked", "N=%d x S=%d = %d folds [STATE]"
                  % (n_show, self.s, n_show * self.s))
        self._row("estimate", "%s over %d containers [DERIVED]"
                  % (fmt_secs(self.gpu_est), self.shards))
        if gated:
            self._row("", "= %.2f USD at %.2f USD per GPU hour [PARAMETER,"
                      " unverified]" % (usd, self.rate * 3600.0))
            self._row("ceiling", "%.2f USD whole effort [POLICY]" % self.ceiling)
            self._row("spent", "%.2f USD %s, leaving %.2f USD [STATE]"
                      % (self.spent or 0.0, self.spent_kind, remaining))
        else:
            self._row("ceiling",
                      "not checked: no verified rate parameter this run")
        if gated:
            buys_s = remaining / self.rate
            _, _, cheapest = run_plan(1, self.s, self.k)
            n_fit = largest_fit(self.rate, remaining, self.s, self.k, self.n)
            if n_fit is None:
                self._row("verdict",
                          "remaining buys %s; the cheapest run is N=1 x S=%d at %s"
                          % (fmt_secs(buys_s), self.s, fmt_secs(cheapest)))
                self._row("options",
                          "raise the ceiling (a policy change), or wait for the"
                          " budget to recover")
            else:
                t_fit, j_fit, g_fit = run_plan(n_fit, self.s, self.k)
                self._row("fits", "up to N=%d (%d folds, %d containers,"
                          " %s) [DERIVED by scan]"
                          % (n_fit, t_fit, j_fit, fmt_secs(g_fit)))
                self._row("options", "rerun with N <= %d" % n_fit)
        else:
            self._row("options",
                      "supply a rate parameter with its source, then resubmit")
        self._row("timings", "load %s, fold %s, measured once;"
                  " the measurement record does not ship"
                  % (fmt_secs(LOAD_S, est=False), fmt_secs(FOLD_S, est=False)))

    def refuse(self, why, options):
        """Event: run refused before acceptance. Nothing was submitted."""
        self._line("REFUSED", self.run_id,
                   "%s, nothing submitted" % why)
        for i, opt in enumerate(options):
            self._row("options" if i == 0 else "", opt)

    def finished(self, folds_ok, shards_ok, gpu_measured=None,
                 wall_measured=None, claims=(), not_claims=(),
                 next_actions=()):
        """Event: run finished. Prints the closing summary block.

        Cost prints dollars only when a verified rate parameter existed this
        run. Otherwise the line says dollars are unavailable. Claims state
        what the landed results support and, just as firmly, what they do not.
        """
        lost = self.total - folds_ok
        head = "%d of %d folds landed" % (folds_ok, self.total)
        if lost:
            head += ", %d lost" % lost
        self._line("FINISHED", self.run_id, head)
        self._row("ran", "%d of %d shards ok" % (shards_ok, self.shards))
        if gpu_measured is None:
            self._row("gpu", "no job reported timings; plan figure was %s"
                      % fmt_secs(self.gpu_est))
        else:
            self._row("gpu", "%s measured from landed jobs; plan figure %s"
                      % (fmt_secs(gpu_measured, est=False),
                         fmt_secs(self.gpu_est)))
        if wall_measured is None:
            self._row("wall", "not recorded; plan range %s"
                      % self._wall_range_text())
        else:
            self._row("wall", "%s measured; plan range %s, success only"
                      % (fmt_min(wall_measured, est=False),
                         self._wall_range_text()))
        if self.rate is None:
            self._row("cost",
                      "dollars unavailable: no verified rate parameter"
                      " this run")
        else:
            self._row("cost", "%.2f USD est at %.2f USD per GPU s"
                      " [PARAMETER, unverified]"
                      % (gpu_measured * self.rate if gpu_measured
                         else self.gpu_est * self.rate, self.rate))
        first_c = True
        for claim in claims:
            self._row("claims" if first_c else "", "supports: %s" % claim)
            first_c = False
        for claim in not_claims:
            self._row("" if first_c else "claims",
                      "does not: %s" % claim)
            first_c = False
        for i, act in enumerate(next_actions):
            self._row("next" if i == 0 else "", act)


class FakeClock(object):
    """Deterministic clock for tests and demos."""

    def __init__(self, start=0.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


# --------------------------------------------------------------------------
# Demo. Builds three transcripts. The design note that displayed them does
# not ship with this package. Timings and metrics
# other than LOAD_S and FOLD_S are illustrative, and every estimate still
# carries its est label. Run: python3 progress.py
# --------------------------------------------------------------------------

def _scenario_refused(p):
    p.run_id = "r-a1"
    p.refuse(
        "tier 'vast' is not configured",
        ["the configured tier is modal",
         "set the tier in the run config, then resubmit"])


def _scenario_gate(p):
    p.configure("r-b1", n_designs=20, seeds=5, shard_size=29, tier="modal",
                concurrency_cap=2,
                rate=4.50 / 3600.0,   # quoted list price, unverified
                ceiling=50.00, spent=49.83)
    p.gate_failed(requested_n=20)


def _scenario_main(p):
    ck = p._clock
    p.configure("r-c1", n_designs=6, seeds=5, shard_size=29, tier="modal",
                concurrency_cap=2, rate=None, ceiling=None, spent=None)
    p.accept()

    ck.advance(1)
    p.submit(1, "j-7f2k", 29)
    p.submit(2, "j-k9dq", 1)

    # Shard 1 lands with the job's own measured timings.
    ck.advance(1192)
    p.landed(1, "j-7f2k", 29, gpu_measured=1186.2, harvest="8.9 MB")
    ck.advance(1)
    demo_metrics = [
        ("d1", [0.938, 0.941, 0.933, 0.945, 0.936]),
        ("d2", [0.912, 0.920, 0.907, 0.918, 0.915]),
        ("d3", [0.884, 0.871, 0.890, 0.877, 0.882]),
        ("d4", [0.846, 0.851, 0.839, 0.855, 0.848]),
        ("d5", [0.802, 0.791, 0.810, 0.798, 0.805]),
    ]
    for name, iptms in demo_metrics:
        p.scored(name, [
            {"seed": i, "iptm": v, "plddt": v - 0.004}
            for i, v in enumerate(iptms)])

    # Shard 2 dies the way fold-A-s0-v2 died: a cold weight read that hit
    # the 1200 s command bound before folding started. That was the attempt
    # before the canary succeeded, and its run log does not ship. Its job
    # started near t+00:01,
    # so the kill lands near t+20:02.
    ck.advance(8)
    p.failed(
        2, "j-k9dq", "fold",
        "killed at the 1200 s command bound during cold weight read",
        "fold step, design d6, all 5 seeds, shard 2 of 2",
        "shard 1 landed intact; 25 of 30 folds are safe on disk",
        ["rerun d6 alone as a 1 fold shard; expect load %s for it"
         % fmt_secs(LOAD_S),
         "do not resubmit shard 1; its outputs are already harvested"])
    ck.advance(1)
    p.finished(
        folds_ok=25, shards_ok=1, gpu_measured=1186.2, wall_measured=1203.0,
        claims=["iptm and plddt ranking for d1 to d5, five seeds each,"
                " frozen formula"],
        not_claims=["anything about d6; its folds never ran, no number for"
                    " d6 exists",
                    "sc_dockq arms need MSA inputs, not run here",
                    "Modal timings; nothing has run on Modal"],
        next_actions=["rerun d6 as its own shard, or pair it with the next"
                      " batch to amortize the load"])


def demo(out=None):
    """Print the three reference transcripts."""
    dst = out if out is not None else sys.stdout
    ck = FakeClock()
    p = Progress(out=dst, clock=ck)

    dst.write("scenario A: refused before acceptance\n")
    _scenario_refused(p)
    dst.write("\nscenario B: budget gate fails\n")
    ck.advance(1)
    _scenario_gate(p)
    dst.write("\nscenario C: a real run, 6 designs x 5 seeds on fal\n")
    p2 = Progress(out=dst, clock=FakeClock())
    _scenario_main(p2)


if __name__ == "__main__":
    demo()

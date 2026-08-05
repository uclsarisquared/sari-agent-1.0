"""One-window pipeline applet: explore -> shelf graph -> audit -> capture -> annotate.

    python slamtest/pipeline_app.py

Runs the whole annotated-map pipeline from a single Tkinter window instead of five terminal
commands. Design decisions, deliberate:

  * STAGES RUN AS SUBPROCESSES OF THE EXISTING CLIs, never as imports. Each stage script owns
    its argparse defaults (step sizes, safety margins, tags); re-declaring any of that here
    would let the two drift apart - the exact trap explore_vlm.py documents. This applet only
    ever passes the few flags the user actually chose; everything else stays whatever the
    stage's own parser says.
  * ONE OUTPUT DIRECTORY FOR THE WHOLE RUN. explore writes grid_final/topology_final there,
    build_shelf_graph derives topology_final_shelf there, capture_walk fills captures/ there,
    annotate_pass writes annotations/products/semantic_map there. The downstream stages' tag
    defaults (final / final_shelf) then just work. For the VLM planner variants, --run-tag is
    pinned to "final" for the same reason - explore_vlm would otherwise name the topology after
    the planner and break the chain.
  * THE FROZEN BASELINE IS HARD-PROTECTED. slamtest/output is the working baseline (CLAUDE.md):
    exploring into it renumbers every checkpoint and invalidates all captures/annotations. The
    Explore stage refuses to target it, full stop - same policy as explore_vlm.run(), enforced
    here too because this window makes re-mapping one click instead of a deliberate command.
  * LIVE MAP = --save-every 1 + polling grid_<step>.png. explore.py already snapshots the grid
    every N steps; at N=1 the newest PNG in the output dir IS the live map, and the applet just
    redraws it twice a second. No IPC, no explore.py changes, and a crashed GUI costs nothing.
    The run is I/O-bound on the Unity scan (~670 ms/step, measured - see memory), so the extra
    per-step PNG write is noise; raise "snapshot every" if it ever matters.
  * ANNOTATOR PIN: the annotate stage defaults to claude -p / sonnet / medium and the fields are
    labelled as pinned (user directive 2026-07-19). The applet does not offer the qwen probe
    path - annotation quality is measured and frozen on sonnet.

Stages 1 (Explore) and 4 (Capture) need the Unity sim in Play mode; the rest are offline.
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

_SLAM_DIR = os.path.dirname(os.path.abspath(__file__))        # slamtest
if _SLAM_DIR not in sys.path:
    sys.path.insert(0, _SLAM_DIR)
import _bootstrap  # noqa: F401,E402  (agent root + all slamtest category dirs)
_OVERHAUL_DIR = os.path.dirname(_SLAM_DIR)
FROZEN_OUTPUT_DIR = os.path.join(_SLAM_DIR, "output")
DEFAULT_RUN_ROOT = os.path.join(_SLAM_DIR, "output_runs")

PLANNERS = ["astar (explore.py)", "astar (instrumented)", "vlm", "vlm-advised", "vlm-goal"]

POLL_MS = 500          # live-map + log poll interval
MAP_PANEL_PX = 520     # max edge of the rendered map preview


class PipelineApp:
    def __init__(self, root):
        self.root = root
        root.title("Sari map pipeline")

        self.proc = None                 # currently running stage subprocess
        self.worker = None
        self.stop_requested = threading.Event()
        self.log_queue = queue.Queue()
        self._last_map_mtime = None
        self._running = False            # True while stages run; _refresh_map picks its rule by this
        self._map_photo = None           # keep a reference or Tk garbage-collects the image

        self._build_ui()
        self._poll()

    # ------------------------------------------------------------------ UI construction

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(top)
        left.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        right = ttk.LabelFrame(top, text="Map (live during Explore)")
        right.grid(row=0, column=1, sticky="nsew")
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        # --- output dir ---
        dirf = ttk.LabelFrame(left, text="Output directory")
        dirf.pack(fill="x", pady=(0, 8))
        self.out_dir = tk.StringVar(value=os.path.join(
            DEFAULT_RUN_ROOT, datetime.now().strftime("run_%m%d_%H%M%S")))
        ttk.Entry(dirf, textvariable=self.out_dir, width=46).grid(
            row=0, column=0, columnspan=3, sticky="we", padx=4, pady=2)
        ttk.Button(dirf, text="Browse...", command=self._browse_dir).grid(row=1, column=0, pady=2)
        ttk.Button(dirf, text="New timestamped", command=self._new_dir).grid(row=1, column=1, pady=2)
        ttk.Button(dirf, text="Open folder", command=self._open_dir).grid(row=1, column=2, pady=2)

        # --- stages ---
        stf = ttk.LabelFrame(left, text="Stages (run in order)")
        stf.pack(fill="x", pady=(0, 8))
        self.stage_vars = {}
        for key, label in [
            ("explore", "1. Explore \U0001F3AE  (builds grid + topology)"),
            ("shelf_graph", "2. Shelf graph  (topology_final_shelf)"),
            ("audit", "3. Reachability audit"),
            ("capture", "4. Capture walk \U0001F3AE"),
            ("annotate", "5. Annotate  (claude -p, offline)"),
        ]:
            v = tk.BooleanVar(value=True)
            self.stage_vars[key] = v
            ttk.Checkbutton(stf, text=label, variable=v).pack(anchor="w")
        ttk.Label(stf, text="\U0001F3AE = Unity sim must be in Play mode",
                  foreground="grey").pack(anchor="w")

        # --- explore options ---
        exf = ttk.LabelFrame(left, text="Explore options")
        exf.pack(fill="x", pady=(0, 8))
        self.planner = tk.StringVar(value=PLANNERS[0])
        self._row(exf, 0, "Planner variant", ttk.Combobox(
            exf, textvariable=self.planner, values=PLANNERS, state="readonly", width=22))
        self.max_steps = tk.StringVar(value="300")
        self._row(exf, 1, "Max steps", ttk.Entry(exf, textvariable=self.max_steps, width=8))
        self.save_every = tk.StringVar(value="1")
        self._row(exf, 2, "Snapshot every N steps", ttk.Entry(exf, textvariable=self.save_every, width=8))
        self.uri = tk.StringVar(value="ws://localhost:8080/commands")
        self._row(exf, 3, "Sim websocket", ttk.Entry(exf, textvariable=self.uri, width=26))

        # --- capture + annotate options ---
        caf = ttk.LabelFrame(left, text="Capture / annotate options")
        caf.pack(fill="x", pady=(0, 8))
        self.angles = tk.StringVar(value="2")
        self._row(caf, 0, "Capture angles (2 = +crouch)", ttk.Combobox(
            caf, textvariable=self.angles, values=["1", "2"], state="readonly", width=5))
        self.limit = tk.StringVar(value="0")
        self._row(caf, 1, "Capture limit (0 = all)", ttk.Entry(caf, textvariable=self.limit, width=8))
        self.ann_backend = tk.StringVar(value="claude-cli")
        self._row(caf, 2, "Annotator backend", ttk.Combobox(
            caf, textvariable=self.ann_backend, values=["claude-cli", "qwen"],
            state="readonly", width=12))
        self.ann_model = tk.StringVar(value="sonnet")
        self._row(caf, 3, "Model (claude only)", ttk.Entry(caf, textvariable=self.ann_model, width=12))
        self.ann_effort = tk.StringVar(value="medium")
        self._row(caf, 4, "Effort (claude only)", ttk.Combobox(
            caf, textvariable=self.ann_effort, values=["low", "medium", "high"], state="readonly", width=8))
        # 0 = annotate_pass's backend default (8 for claude-cli, 4 for qwen); 1 = old sequential.
        self.ann_jobs = tk.StringVar(value="0")
        self._row(caf, 5, "Parallel jobs (0 = backend default)", ttk.Entry(caf, textvariable=self.ann_jobs, width=8))
        # Fused stages 4+5 (capture_annotate_walk.py): annotate each checkpoint while the walk
        # drives to the next. Only takes effect when BOTH stages are ticked; off by default so the
        # baseline sequential behaviour is unchanged.
        self.fuse_cap_ann = tk.BooleanVar(value=False)
        ttk.Checkbutton(caf, text="Fuse capture+annotate (annotate during the walk)",
                        variable=self.fuse_cap_ann).grid(row=6, column=0, columnspan=2, sticky="w")
        ttk.Label(caf, text="claude-cli = measured baseline (sonnet/medium). qwen = self-hosted,\n"
                            "no subscription; model/effort fields ignored. A/B before trusting qwen.\n"
                            "Jobs default: 8 claude-cli, 4 qwen. Set 1 if rate limits bite.\n"
                            "Fused runs as ONE stage; needs both stage 4 and 5 ticked.",
                  foreground="grey").grid(row=7, column=0, columnspan=2, sticky="w")

        # --- run controls ---
        ctl = ttk.Frame(left)
        ctl.pack(fill="x", pady=(0, 8))
        self.run_btn = ttk.Button(ctl, text="Run pipeline", command=self.start)
        self.run_btn.pack(side="left", padx=(0, 6))
        self.stop_btn = ttk.Button(ctl, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left")
        self.status = tk.StringVar(value="idle")
        ttk.Label(ctl, textvariable=self.status).pack(side="left", padx=10)

        # --- map panel ---
        self.map_label = ttk.Label(right, text="(no map yet)", anchor="center")
        self.map_label.pack(fill="both", expand=True, padx=4, pady=4)
        self.map_caption = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.map_caption, foreground="grey").pack(anchor="w", padx=4)

        # --- log ---
        logf = ttk.LabelFrame(self.root, text="Log")
        logf.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.root.rowconfigure(1, weight=1)
        self.log = tk.Text(logf, height=14, wrap="none", state="disabled",
                           bg="#111", fg="#ddd", font=("Consolas", 9))
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(logf, command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

    @staticmethod
    def _row(parent, r, label, widget):
        ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w", padx=4, pady=1)
        widget.grid(row=r, column=1, sticky="w", padx=4, pady=1)

    # ------------------------------------------------------------------ dir helpers

    def _browse_dir(self):
        d = filedialog.askdirectory(initialdir=os.path.dirname(self.out_dir.get()) or _SLAM_DIR)
        if d:
            self.out_dir.set(d)

    def _new_dir(self):
        self.out_dir.set(os.path.join(DEFAULT_RUN_ROOT,
                                      datetime.now().strftime("run_%m%d_%H%M%S")))

    def _open_dir(self):
        d = self.out_dir.get()
        os.makedirs(d, exist_ok=True)
        os.startfile(d)  # Windows; this project is Windows-only (winsound history)

    # ------------------------------------------------------------------ pipeline

    def _commands(self):
        """Build the stage command lines from the UI state. Only user-chosen flags are passed;
        every other default stays owned by the stage's own parser."""
        out = self.out_dir.get()
        py = [sys.executable, "-u"]  # -u: unbuffered, so the log streams line by line
        cmds = []

        if self.stage_vars["explore"].get():
            planner = self.planner.get()
            if planner == PLANNERS[0]:
                cmd = py + [os.path.join(_SLAM_DIR, "drivers", "explore.py"),
                            "--output-dir", out]
            else:
                name = "astar" if planner == PLANNERS[1] else planner
                cmd = py + [os.path.join(_SLAM_DIR, "drivers", "explore_vlm.py"),
                            "--planner", name,
                            "--output-dir", out,
                            # Pin the tag so topology lands as topology_final.json and the
                            # downstream stages' defaults chain without retag flags.
                            "--run-tag", "final"]
            cmd += ["--max-steps", self.max_steps.get(),
                    "--save-every", self.save_every.get(),
                    "--uri", self.uri.get()]
            cmds.append(("Explore", cmd))

        if self.stage_vars["shelf_graph"].get():
            cmds.append(("Shelf graph", py + [os.path.join(_SLAM_DIR, "graph", "build_shelf_graph.py"), out]))

        if self.stage_vars["audit"].get():
            cmds.append(("Audit", py + [os.path.join(_SLAM_DIR, "graph", "audit_standability.py"),
                                        out, "--topology-tag", "final_shelf"]))

        def annotator_flags():
            backend = self.ann_backend.get()
            flags = ["--backend", backend]
            if backend == "claude-cli":
                # model/effort apply to claude only; qwen uses its own default ($SARI_ANNOTATOR_MODEL).
                flags += ["--model", self.ann_model.get(), "--effort", self.ann_effort.get()]
            jobs = self.ann_jobs.get().strip()
            if jobs and jobs != "0":  # 0/blank = let annotate_pass pick the backend default (8/4)
                flags += ["--jobs", jobs]
            return flags

        # Fusing replaces stages 4 and 5 with ONE command (capture_annotate_walk.py) that
        # annotates each checkpoint while walking to the next. Only when both stages are ticked -
        # a lone capture or lone annotate has nothing to fuse with.
        fuse = (self.fuse_cap_ann.get() and self.stage_vars["capture"].get()
                and self.stage_vars["annotate"].get())
        if fuse:
            cmds.append(("Capture+Annotate", py +
                         [os.path.join(_SLAM_DIR, "capture", "capture_annotate_walk.py"), out,
                          "--limit", self.limit.get(),
                          "--angles", self.angles.get(),
                          "--uri", self.uri.get()] + annotator_flags()))
        else:
            if self.stage_vars["capture"].get():
                cmds.append(("Capture", py + [os.path.join(_SLAM_DIR, "capture", "capture_walk.py"), out,
                                              "--limit", self.limit.get(),
                                              "--angles", self.angles.get(),
                                              "--uri", self.uri.get()]))

            if self.stage_vars["annotate"].get():
                cmds.append(("Annotate", py + [os.path.join(_SLAM_DIR, "annotate", "annotate_pass.py"),
                                               out] + annotator_flags()))
        return cmds

    def start(self):
        out = os.path.abspath(self.out_dir.get())
        if self.stage_vars["explore"].get() and out == os.path.abspath(FROZEN_OUTPUT_DIR):
            # Same policy as explore_vlm.run(): the frozen map is the working baseline, and
            # re-exploring it renumbers every checkpoint (CLAUDE.md). Not overridable here.
            messagebox.showerror(
                "Protected directory",
                "That is the FROZEN baseline map (slamtest/output).\n\n"
                "Exploring into it would wipe it and renumber every checkpoint, invalidating "
                "all captures and annotations keyed to the old ids.\n\n"
                "Pick a different output directory (e.g. the timestamped default).")
            return
        cmds = self._commands()
        if not cmds:
            messagebox.showinfo("Nothing to do", "No stages selected.")
            return
        needs_sim = [n for n, _ in cmds if n in ("Explore", "Capture", "Capture+Annotate")]
        if needs_sim and not messagebox.askokcancel(
                "Sim check", f"{' and '.join(needs_sim)} need the Unity sim in Play mode.\n"
                             f"Is it running?"):
            return

        os.makedirs(out, exist_ok=True)
        self.stop_requested.clear()
        self._last_map_mtime = None
        self._running = True   # _refresh_map: newest-PNG (live) mode while stages run
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.worker = threading.Thread(target=self._run_stages, args=(cmds,), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_requested.set()
        p = self.proc
        if p and p.poll() is None:
            p.terminate()
        self._log("\n[pipeline] STOP requested - terminating current stage.\n")

    def _run_stages(self, cmds):
        for i, (name, cmd) in enumerate(cmds, 1):
            if self.stop_requested.is_set():
                break
            self._set_status(f"{name} ({i}/{len(cmds)})")
            self._log(f"\n===== [{i}/{len(cmds)}] {name} =====\n$ {subprocess.list2cmdline(cmd)}\n")
            t0 = time.monotonic()
            try:
                self.proc = subprocess.Popen(
                    cmd, cwd=_OVERHAUL_DIR, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            except OSError as e:
                self._log(f"[pipeline] failed to launch {name}: {e}\n")
                break
            for line in self.proc.stdout:
                self._log(line)
            rc = self.proc.wait()
            self.proc = None
            elapsed = time.monotonic() - t0
            if self.stop_requested.is_set():
                self._log(f"[pipeline] {name} stopped by user after {elapsed:.1f}s.\n")
                break
            if rc != 0:
                # A failed stage invalidates everything downstream of it - halt rather than
                # annotate a map that never finished building.
                self._log(f"[pipeline] {name} exited with code {rc} after {elapsed:.1f}s - halting pipeline.\n")
                break
            self._log(f"[pipeline] {name} finished in {elapsed:.1f}s.\n")
        else:
            self._log("\n[pipeline] all selected stages complete.\n")
        self._set_status("idle")
        self.log_queue.put(("__controls__", None))

    # ------------------------------------------------------------------ GUI-thread plumbing

    def _log(self, text):
        self.log_queue.put(("log", text))

    def _set_status(self, text):
        self.log_queue.put(("status", text))

    def _poll(self):
        # Drain the log queue (worker thread -> GUI thread; Tk is not thread-safe).
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self.log.configure(state="normal")
                    self.log.insert("end", payload)
                    self.log.see("end")
                    self.log.configure(state="disabled")
                elif kind == "status":
                    self.status.set(payload)
                elif kind == "__controls__":
                    self.run_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self._running = False        # settle the map panel on the shelf graph
                    self._last_map_mtime = None  # force one re-pick under the idle rule
        except queue.Empty:
            pass
        self._refresh_map()
        self.root.after(POLL_MS, self._poll)

    def _refresh_map(self):
        """Map panel source, by pipeline state:
          * RUNNING: newest grid_*.png or shelf_graph_*.png wins. With snapshot-every=1 the grid
            snapshots are the live map during Explore, and the moment the shelf-graph stage writes
            its PNG that becomes the newest, so the panel upgrades mid-run.
          * IDLE: shelf_graph_*.png wins WHENEVER IT EXISTS, else newest grid. Presence beats
            mtime here on purpose: the graph (checkpoints + edges over the same geometry) is the
            informative final view, and mtimes can lie when idle - the frozen baseline's
            grid_final.png is literally newer on disk than its shelf graph."""
        d = self.out_dir.get()
        if not os.path.isdir(d):
            return
        newest, newest_m = None, -1
        try:
            candidates = [fn for fn in os.listdir(d)
                          if (fn.startswith("grid_") or fn.startswith("shelf_graph_"))
                          and fn.endswith(".png")]
            if not self._running:
                graphs = [fn for fn in candidates if fn.startswith("shelf_graph_")]
                candidates = graphs or candidates
            for fn in candidates:
                m = os.path.getmtime(os.path.join(d, fn))
                if m > newest_m:
                    newest, newest_m = fn, m
        except OSError:
            return
        if newest is None or newest_m == self._last_map_mtime:
            return
        path = os.path.join(d, newest)
        try:
            from PIL import Image, ImageTk
            im = Image.open(path)
            im.thumbnail((MAP_PANEL_PX, MAP_PANEL_PX))
            self._map_photo = ImageTk.PhotoImage(im)
            self.map_label.configure(image=self._map_photo, text="")
            self.map_caption.set(f"{newest}  ({datetime.fromtimestamp(newest_m):%H:%M:%S})")
            self._last_map_mtime = newest_m
        except OSError:
            # Snapshot may still be mid-write at save-every=1; the next poll gets it whole.
            pass


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    PipelineApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

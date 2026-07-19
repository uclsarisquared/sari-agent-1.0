# Hard-coded aids vs VLM-based aids — literature review and defense brief

**Question.** Should the Sari agent's spatial aids (LiDAR mapping, occupancy grid, A*, checkpoint/shelf
graph) be deterministic code, or should the aids themselves be LLM/VLM-based — up to a fully
autonomous VLM spatial navigator, as the adviser argues the thesis requires?

**How this document was produced (2026-07-17).** A multi-agent deep-research pass: 5 parallel search
angles → source fetching → **every headline claim below verified against its primary source**, the
core set by 3 independent adversarial verifiers each instructed to *refute* the claim (24 claims
survived 3–0; 1 was refuted and is listed in §9 as do-not-use). A second pass verified 25 additional
citations (venue + verbatim quotes) by fetching each arXiv/proceedings page. Codebase grounding
comes from a full inventory of `overhaul/`, `slamtest/`, and `SariSandboxV2`. Verification status is
marked per entry in §10.

---

## 0. TL;DR — the answer

**The dividing line the literature draws is by *task type*, not by implementation fashion:**

- **Metric–geometric functions** (localization, mapping, collision, path search) → **deterministic /
  specialist modules.** VLMs measurably lack these primitives (six independent peer-reviewed
  benchmarks, 20–40 points below human, §2); prompting techniques make spatial performance *worse*
  (§2.2); autoregressive models cannot do sound planning or self-verification (§5.2); and the only
  large-scale real-world modular-vs-end-to-end study found 90% vs 23% success in favor of the
  map-based modular pipeline (§3).
- **Semantic functions** (open-vocabulary perception, label reading, goal interpretation, task
  decomposition) → **VLM/LLM.** This is where foundation models earn their keep, and it is exactly
  where the Sari pipeline already uses one (the shelf annotator: ~97–98% precision when scoped to
  "what is in front of you", `CLAUDE.md:85-88`).

Every celebrated LLM/VLM-robotics paper of 2022–2024 — SayCan, LM-Nav, VLMaps, ConceptGraphs,
VoxPoser, SayPlan, NLMap, OK-Robot, Mobility VLA, CoW (§4) — draws the line in the same place: the
L(V)LM never owns metric spatial truth; a graph, map, value function, or classical planner does.
The Sari architecture ("the graph owns spatial truth; the VLM only judges what is directly in front
of it; the agent verifies on arrival", `CLAUDE.md:14-16`) is not a workaround of this literature —
it is an instance of its consensus, with a citable name: an **LLM-Modulo architecture**
(Kambhampati et al., ICML 2024, §5.2).

**And the "VLM-based aids" alternative is not hypothetical here — it was the old overhaul agent**
(orchestrator LLM + LLM observer/mode-router + LLM episodic reflector + Gemini/Moondream detectors,
with the main VLM doing raw path planning and degree arithmetic from a first-person view,
`sys_inst.py:129`, `agent.py:216`). Its navigation failure is the documented reason this redesign
exists (`NavReasonPlan.md:5`). Splitting the work across *more* VLMs diversifies style, not
capability class: the missing competency is spatial, and every model in that stack is missing it.

**On "cheating" (§7):** the field's founding benchmarks hand agents far stronger aids than Sari's —
original VLN moves agents on a **hand-built navigation graph** with "known environment topologies,
short-range oracle navigation, and perfect agent localization" (VLN-CE's own characterization), and
canonical ObjectNav equips agents with an **idealized GPS+Compass sensor**. What the field treats as
illegitimate is *privileged information* — oracle access the deployed system couldn't have. Sari's
map is built by the agent itself, from its own onboard LiDAR/RGB, through the same WebSocket API it
acts through; the sim's privileged-query stubs are unwired (`env.py` `RequestAnnotation`/
`RequestJson`). Self-acquired structure is autonomy, not a violation of it.

**The honest caveats (§6, §9):** end-to-end VLM navigators do exist (NaVid ~66% real-world VLN;
PoliFormer 85.5% sim ObjectNav) — but they are trained on 0.5M–3.6M navigation trajectories or
hundreds of millions of RL interactions, several are themselves modular or distilled *from classical
planners*, and none addresses the measured zero-shot spatial deficits. Two numbers currently in
`VMap_Plan.md` are misattributed and must be corrected before any committee member checks them.

---

## 1. The question, stated precisely

Three candidate architectures:

| | Architecture | Status in this project |
|---|---|---|
| A | **End-to-end VLM navigator** — the reasoner VLM perceives, plans, and moves | The old overhaul's navigation mode; documented as the primary failure mode (`NavReasonPlan.md:5`) |
| B | **VLM + VLM-based aids** — separate L(V)LM subagents assist (observer, reflector, detector, navigator-LLM) | The old overhaul as shipped: 3 LLM calls/step + Gemini/Moondream/DA3 (`subtask_agents.py`, `agent.py:216`, `perception.py`) |
| C | **VLM + deterministic aids** — LiDAR SLAM, A*, checkpoint graph own geometry; VLM scoped to semantics | `slamtest/` pipeline; the plan under dispute |

The adviser's objection is that C "cheats" the goal of a fully autonomous VLM spatial navigator
(A/B). The evidence below is organized as: VLMs lack the spatial primitives (§2); modular beats
end-to-end where it has been tested at scale (§3); the field's flagship agent papers all use C (§4);
when the aid itself is an L(V)LM it measurably underperforms, including in this repo (§5); the
honest counter-evidence (§6); and why C is not cheating by the field's own standards (§7).

---

## 2. Pillar 1 — VLMs measurably lack the spatial primitives navigation requires

All claims in this section passed 3–0 adversarial verification against the primary papers.

### 2.1 The benchmark record (2024–2026)

| Benchmark (venue) | What it tests | Best model | Humans | Link |
|---|---|---|---|---|
| **VSI-Bench** (CVPR 2025) | Visual-spatial intelligence from video (distances, sizes, layout, route plans) | Gemini-1.5 Pro **45.4%** (GPT-4o 34.0%) | **79%** | [arXiv:2412.14171](https://arxiv.org/abs/2412.14171) |
| **BLINK** (ECCV 2024) | Core visual perception: relative depth, correspondence, multi-view | GPT-4V **51.26%**, Gemini 45.72% (random 38.09%) | **95.70%** | [arXiv:2404.12390](https://arxiv.org/abs/2404.12390) |
| **BlindTest** (ACCV 2024 oral) | Trivial 2D geometry (do circles overlap, count line crossings) | 4-model avg **58.07%**, best 77.84% | ~100% (expected) | [arXiv:2407.06581](https://arxiv.org/abs/2407.06581) |
| **SpatialVLM eval** (CVPR 2024) | Quantitative metric estimation from images | GPT-4V **0.0%** within [50%, 200%] of true distance; best baseline 33.9% | — | [arXiv:2401.12168](https://arxiv.org/abs/2401.12168) |
| **ViewSpatial-Bench** (2025) | Cross-viewpoint localization | GPT-4o **34.98%** vs random 26.33% ("barely outperforming random chance") | — | [arXiv:2505.21500](https://arxiv.org/abs/2505.21500) |
| **MV-RoboBench** (ICLR 2026) | Multi-view spatial reasoning in real robot scenes | GPT-5 **56.41%** (random ~19.8%) | **91.0%** | [arXiv:2510.19400](https://arxiv.org/abs/2510.19400) |
| **SPACE** (ICLR 2025) | Classic animal-cognition spatial tests | Abstract's own opening answer: **"Not yet."** — frontier models "performing near chance level" | — | [arXiv:2410.06468](https://arxiv.org/abs/2410.06468) |

Load-bearing details behind the table:

- **VSI-Bench error analysis:** ~71% of the best model's errors are *spatial reasoning* errors
  (relational reasoning, egocentric–allocentric transformation) — not perception, not language.
  "Spatial reasoning is the primary bottleneck for MLLM performance on VSI-Bench."
- **BLINK's worst tasks are SLAM primitives** (interpretive bridge — ours, but the numbers are the
  paper's): relative depth GPT-4V 58.87% vs human 99.19; visual correspondence 37.21 vs 99.42;
  multi-view reasoning 58.65 vs 92.48, with Gemini *below random* on multi-view.
- **MV-RoboBench transfer failure:** models strong on single-view spatial benchmarks (OmniSpatial)
  sit near the ~19.8% random baseline on multi-view versions of the same competencies — single-image
  skill does not compose into the embodied multi-view regime, which is precisely the regime a
  first-person navigator lives in.
- **Persistence:** the gaps narrow but do not close through 2026 (BLINK best ~81.4% vs 95.7 human on
  the July 2026 leaderboard; MV-RoboBench evaluates GPT-5-class models). Date-qualify all numbers to
  their model generation when citing (§9).

### 2.2 Prompting does not fix it — it makes it worse

On VSI-Bench, the three standard reasoning techniques all *degrade* spatial performance:
chain-of-thought **−4%**, self-consistency **−1.1%**, tree-of-thoughts **−4%** — while the identical
CoT prompt *improves* the same model's general video QA (VideoMME 77.2 → 79.8). The deficit is not
an elicitation problem. ([arXiv:2412.14171](https://arxiv.org/abs/2412.14171), verified 3–0.)

This has a local replica: this repo's Phase-4.1 probe found Qwen3.6 thinking mode "spent its whole
budget thinking, looped, and never answered" (`explore_vlm.py:160-162`), and the annotator found the
same for annotation (`annotator_sys_inst.py`, MEASURED block).

### 2.3 What *does* move the needle: external spatial structure

The VSI-Bench ablation is the cleanest published version of exactly this thesis dispute: prompting
the model to build a "cognitive map" raises relative-distance accuracy **46.0% → 56.0%**, and
supplying a **ground-truth external map raises it to 66.0%**. An accurate externally-built map is
worth +20 points where in-context spatial reasoning stalls — and *building accurate external maps
from range sensors is a solved deterministic problem* (that bridge is ours; the numbers are the
paper's; scope caveat: one model, the relative-distance subtask — see §9).

The benchmark authors' own prescriptions point the same way. MV-RoboBench (ICLR 2026, verified
verbatim in both arXiv v1 and the camera-ready): progress requires "**architectures that explicitly
encode geometric priors and enforce cross-view consistency**", and "**scaling perception alone is
insufficient — models require explicit reasoning mechanisms to transform multi-view observations
into actionable, embodied understanding.**" BLINK: "specialist CV models could solve these problems
much better" — specialists beat generalist VLMs by 18–57% per task. Even the strongest
vendor-reported fix concedes the point: o3 reaches ~90% on BlindTest only *with image-manipulation
tools* (non-peer-reviewed, OpenAI-reported) — i.e., the frontier lab's own remedy for VLM spatial
failure is *giving the model tools*.

---

## 3. Pillar 2 — where modular vs end-to-end has been tested at scale, modular won

**Gervet, Chintala, Batra, Malik, Chaplot — "Navigating to Objects in the Real World", Science
Robotics 2023** ([arXiv:2212.00922](https://arxiv.org/abs/2212.00922),
[DOI 10.1126/scirobotics.adf6991](https://www.science.org/doi/10.1126/scirobotics.adf6991)) — the
single most on-point citation for this defense; every sentence below verified verbatim, 3–0:

- The **modular** pipeline (classical geometric map + planner, enriched with learned semantic
  perception/exploration) achieved **90% ObjectNav success across six unseen real homes**, no prior
  maps.
- The state-of-the-art **end-to-end** learned policy **collapsed from 77% in simulation to 23% in
  the real world** "due to a large image domain gap between simulation and reality."
- The paper's stated practitioner takeaway: "**modular learning is a reliable approach to navigate
  to objects: modularity and abstraction in policy design enable Sim-to-Real transfer**" — the map
  layer insulates planning from raw pixels because the semantic map space is invariant between
  simulation and reality.
- Scope caveat (keep it honest): six homes, six goal categories, one representative 2022-era
  end-to-end policy (§6 has the later counter-examples).

Supporting precedent, verified:

- **SemExp** (NeurIPS 2020, [arXiv:2007.00643](https://arxiv.org/abs/2007.00643)) — modular
  episodic semantic map + Fast Marching Method planner; **won the CVPR 2020 Habitat ObjectNav
  Challenge** (25.3% vs 18.8% for second place); 13/20 on a real robot.
- **Active Neural SLAM** (ICLR 2020, [arXiv:2004.05155](https://arxiv.org/abs/2004.05155)) —
  learned SLAM/global-policy modules around an **analytical local planner**; won the CVPR 2019
  Habitat PointGoal Challenge.
- **Cognitive Mapping and Planning** (CVPR 2017, [arXiv:1702.03920](https://arxiv.org/abs/1702.03920))
  — the map+planner decomposition predates LLMs entirely; note its planner is differentiable, so
  cite it as precedent for the *decomposition*, not for classical planners specifically.

Note the pattern in the challenge winners: learning is used *inside* modules (semantic segmentation,
where-to-look policies) while the geometry/search layer stays analytical. "Modular" does not mean
"no learning" — it means each function is implemented by the class of system that is actually good
at it. That is the same ledger Sari uses (Depth-Anything in SariVoxeLLMap and the VLM annotator are
learned specialists; the grid and A* are analytical).

---

## 4. Pillar 3 — the celebrated LLM/VLM-agent papers all delegate spatial truth to structure

All ten verified against primary sources (links + venues confirmed; quotes verbatim from abstracts
or bodies as noted). None of these systems lets the language model own metric geometry — and none
was accused of "cheating"; several are among the most-cited robotics papers of their years.

| System (venue) | The L(V)LM does | The deterministic/structural aid | Link |
|---|---|---|---|
| **SayCan** (CoRL 2022) | high-level semantic knowledge | **value functions** "provide the grounding necessary"; skills execute; 84%/74% plan/exec on 101 real instructions | [arXiv:2204.01691](https://arxiv.org/abs/2204.01691) |
| **LM-Nav** (CoRL 2022) | GPT-3 "only to decode the instructions into textual landmarks"; CLIP grounds them | VNM builds a **topological graph** ("mental map") and plans over it | [arXiv:2207.04429](https://arxiv.org/abs/2207.04429) |
| **VLMaps** (ICRA 2023) | open-vocab queries | visual-language features fused into a **3D reconstruction from RGB-D SLAM**; 59% vs CoW 42% / LM-Nav 26% on single subgoals | [arXiv:2210.05714](https://arxiv.org/abs/2210.05714) |
| **ConceptGraphs** (ICRA 2024) | queries/plans over the graph | deterministic **open-vocab 3D scene graph** via multi-view fusion | [arXiv:2309.16650](https://arxiv.org/abs/2309.16650) |
| **VoxPoser** (CoRL 2023 oral) | writes code composing value maps | **3D value maps executed by a model-based planner** | [arXiv:2307.05973](https://arxiv.org/abs/2307.05973) |
| **NLMap** (ICRA 2023) | context-conditioned planning | **queryable scene representation** built before planning | [arXiv:2209.09874](https://arxiv.org/abs/2209.09874) |
| **SayPlan** (CoRL 2023 oral) | semantic search over a collapsed graph | **3D scene graph + "integrating a classical path planner"** + replanning against a scene-graph simulator; 3 floors/36 rooms/140 assets | [arXiv:2307.06135](https://arxiv.org/abs/2307.06135) |
| **OK-Robot** (RSS 2024) | open-vocab object queries | **VoxelMap + A\* navigation** + AnyGrasp; 58.5% across 10 real homes (82% uncluttered), ~1.8× prior work | [arXiv:2401.12202](https://arxiv.org/abs/2401.12202) |
| **Mobility VLA** (CoRL 2024, DeepMind) | long-context Gemini finds the goal frame in a tour video | low-level actions from an **offline-constructed topological graph**; 86%/90% on 57 instructions, 836 m² | [arXiv:2407.07775](https://arxiv.org/abs/2407.07775) |
| **CoW** (CVPR 2023) | CLIP localizes the goal object | **classical frontier-based exploration**, no training; matches a ZSON method trained 500M steps; +15.6 pts on RoboTHOR | [arXiv:2203.10421](https://arxiv.org/abs/2203.10421) |

Two of these deserve a highlight in the defense:

- **LM-Nav is the exact template for Sari's design**: language model for semantics, graph for space,
  classical search for routes — at CoRL, from the group (Levine et al.) that champions learning.
- **Mobility VLA is DeepMind in 2024** doing "frontier VLM + topological graph" — with the graph,
  not the VLM, generating every low-level action. If deterministic scaffolding were cheating,
  DeepMind cheated at CoRL last year.

---

## 5. Pillar 4 — what happens when the aid itself is an L(V)LM

### 5.1 Measured: language-model navigators underperform, and the good ones import maps anyway

- **NavGPT** (AAAI 2024, [arXiv:2305.16986](https://arxiv.org/abs/2305.16986)) — the purest test of
  "let GPT-4 navigate, zero-shot, with explicit reasoning": R2R val-unseen **SR 34% vs 72% for
  trained DUET** (paper table; the abstract itself concedes performance "still falling short of
  trained models"). This is architecture A/B run by its strongest proponents.
- **MapGPT** (ACL 2024, [arXiv:2401.07314](https://arxiv.org/abs/2401.07314)) — the way LLM-driven
  VLN got better was to **inject an online topological map** ("node information and topological
  relationships") into the prompt with map-based multi-step planning (~10–12% SR gains). I.e., the
  LLM-navigation literature's own best practice is to hand the LLM the very scaffold under dispute.
- **InstructNav** (CoRL 2024, [arXiv:2406.04882](https://arxiv.org/abs/2406.04882)) — zero-shot
  generic instruction navigation *via* "Multi-sourced Value Maps" so that linguistic planning "can
  be converted into robot actionable trajectories" — again: language proposes, structure disposes.

### 5.2 Why: autoregressive models cannot do sound planning — the Kambhampati line

- **PlanBench** (NeurIPS 2023 D&B, [arXiv:2206.10498](https://arxiv.org/abs/2206.10498)): "on many
  critical capabilities — including plan generation — LLM performance falls quite short." GPT-4:
  **34.6%** zero-shot on 600-instance Blocksworld, collapsing to **0.16–4.3%** on the semantically
  identical but obfuscated Mystery Blocksworld (numbers re-reported in
  [arXiv:2409.13373](https://arxiv.org/abs/2409.13373), Table 1) — i.e., what looks like planning is
  largely surface-pattern retrieval.
- **Valmeekam et al.** (NeurIPS 2023 Spotlight, [arXiv:2305.15771](https://arxiv.org/abs/2305.15771)):
  "LLMs' ability to generate executable plans autonomously is rather limited, with the best model
  (GPT-4) having an average success rate of ~12% across the domains" (autonomous mode, cross-domain
  average — phrase it exactly this way, §9).
- **LLM-Modulo** (ICML 2024 position paper, [arXiv:2402.01817](https://arxiv.org/abs/2402.01817)):
  "**auto-regressive LLMs cannot, by themselves, do planning or self-verification**"; they are best
  used as "universal approximate knowledge sources" inside frameworks that "combine the strengths of
  LLMs with **external model-based verifiers**." **This is the citable name for Sari's architecture**:
  the VLM proposes semantics; the grid/A*/graph is the external sound verifier-planner; the agent
  verifies on arrival. Even the standability audit (`audit_standability.py`) and the
  validate-never-correct rule in the Phase-4.1 harness (`vlm_planner.py:601-659`) are textbook
  LLM-Modulo moves.

### 5.3 This repo already ran the VLM-as-aid experiment — twice

- **Architecture B at full scale (the old overhaul).** Orchestrator LLM (`subtask_agents.py:48`),
  LLM semantic observer/mode-router + LLM episodic reflector (~3 LLM calls per step,
  `agent.py:216`), Gemini bounding-box centering, Moondream detection, PaddleOCR, Replicate depth —
  and the main VLM still had to do global path planning and degree arithmetic from a first-person
  view ("349.46 + 4*(2.5) = 359.46", `sys_inst.py:129`). Outcome, as documented:
  > "Navigation is the primary failure mode of the agent: it collides with walls, takes suboptimal
  > paths, and the VLM wastes its reasoning budget on global path planning it fundamentally cannot
  > do from a first-person view." (`NavReasonPlan.md:5`)
  Adding more VLM aids did not help because none of them supplied the missing competency class.
- **The controlled version (Phase 4.1).** The ablation harness gives the VLM planner a strict
  *superset* of A*'s information — the same occupancy grid as a top-down image *and* as ASCII text
  (immune to vision-encoder downscale), every frontier's exact world coordinates/distance/bearing,
  its own pose, its blocked-proposal history, plus the first-person camera A* cannot use — and
  validates but never corrects its waypoints (`phase4.1_navigation_ablation.md:34-49`,
  `vlm_planner.py:18-44`). The one offline pilot so far: **8/8 identical waypoints straight through
  a shelf face plainly present as `##########` in the ASCII map it was handed**, with stated
  reasoning "The path directly in front of the robot is clear of obstacles (white floor on the
  map)"; A* solved the identical instance in 5 waypoints around the shelf's west end
  (`phase4.1_navigation_ablation.md:120-135`). One pose, one map, offline — *not a rate* — but a
  reproducible demonstration that the failure is the model class, not information starvation.

### 5.4 The honest nuance: where a VLM aid *is* defensible

Semantic *goal selection* is a different question from geometric *path execution*. SemExp won its
challenge by learning *where to look* while classical code did the moving; the Phase-4.1 harness's
`vlm-goal` arm (VLM picks the frontier, A* plans the path, `explore_vlm.py:120-127`) is exactly that
split and is worth measuring seriously — semantic priors ("juice is probably near dairy") are the
kind of knowledge VLMs actually have. If the adviser wants a VLM-based aid with a real chance of
*beating* the deterministic baseline, `vlm-goal` is it — and the harness to test it fairly already
exists. That is a much stronger story than VLM waypoint generation, which both the literature (§5.1)
and the pilot (§5.3) show failing.

---

## 6. Counter-evidence, stated honestly (and why it doesn't rescue architecture A/B here)

These papers show end-to-end (or VLM-centric) navigation *can* work. All verified against primary
sources; do not omit them — engaging them is what makes the defense credible.

- **NaVid** (RSS 2024, [arXiv:2402.15852](https://arxiv.org/abs/2402.15852)): video-only VLN — "without
  any maps, odometers, or depth inputs" — R2R-CE val-unseen SR 37.4/SPL 35.9, and **~66% success on
  200 real-world instructions**. Trained on **510k navigation samples + 763k web samples**.
- **NaVILA** (RSS 2025, [arXiv:2412.04453](https://arxiv.org/abs/2412.04453)): R2R-CE SR 54.0 (vs
  NaVid 37.0), 88% real-world on 25 instructions — but note: it is itself a **2-level modular
  system** (VLA emits mid-level language actions like "moving forward 75cm"; a separate RL
  locomotion policy executes).
- **Uni-NaVid** (RSS 2025, [arXiv:2412.06224](https://arxiv.org/abs/2412.06224)): unifies VLN/
  ObjectNav/EQA/following; **3.6M navigation samples**.
- **SPOC** (CVPR 2024, [arXiv:2312.02976](https://arxiv.org/abs/2312.02976)): RGB-only real-world
  navigation/manipulation — trained by **imitating shortest-path planners** across ~200,000
  procedurally generated houses. The end-to-end policy is a distillation *of a classical planner*.
- **PoliFormer** (CoRL 2024 Outstanding Paper, [arXiv:2406.20083](https://arxiv.org/abs/2406.20083)):
  **85.5% ObjectNav on CHORES-S** (+28.5 absolute), real-world transfer without adaptation — via
  on-policy RL for "hundreds of millions of interactions."

Why this does not support "the Sari reasoner VLM should navigate":

1. **Every one of these is a trained specialist**, built on 10⁵–10⁸ navigation trajectories/
   interactions. None is a zero-shot frontier VLM asked to navigate — which is the actual
   architecture A/B proposal and the thing the benchmarks in §2 measure. The thesis has no
   navigation-training corpus, no reward pipeline, and no GPU fleet; "train NaVid for the sari-sari
   store" is a different (and much larger) thesis.
2. **Architecturally, they concede the point.** NaVILA splits high-level VLA from low-level control;
   SPOC's teacher is a shortest-path planner; NaVid is a *dedicated navigation model* distinct from
   any task reasoner. Navigation is delegated to a specialist in every case — the debate collapses
   to *classical specialist vs learned specialist*, not *specialist vs none*. For a thesis without
   training infrastructure, the classical specialist is the reproducible, verifiable choice (A*'s
   `crosses_obstacle_rate` is 0 *by construction*, `nav_metrics.py:104-107`).
3. **Real-world reliability still favors the modular stack** where compared at scale (90% vs 23%,
   §3); NaVid's ~66% real-world is impressive and still 24 points below it.

**The adviser's strongest general argument** is Sutton's *Bitter Lesson*
([incompleteideas.net/IncIdeas/BitterLesson.html](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)):
"general methods that leverage computation are ultimately the most effective, and by a large
margin." Three honest responses: (1) the bitter lesson's engine is *learning at scale on the task*
— a zero-shot thesis cannot ride that curve, and the papers that do (SPOC, PoliFormer) had to spend
compute the project does not have; (2) where the scaled methods succeed at spatial tasks today they
do it *through* structure — SPOC distills a planner, o3 fixes BlindTest with image tools, MapGPT
injects maps; (3) the Sari design is bitter-lesson-compatible: the VLM-facing interface (a text/
graph map, per-shelf images) is model-agnostic, so every future stronger VLM slots in without
touching the deterministic layer — and Phase 4.1 exists precisely to re-measure whether the scaffold
is still needed as models improve. The scaffold is a *falsifiable, dated* engineering decision, not
an article of faith.

---

## 7. The "cheating" objection, addressed on the field's own terms

**1. The founding benchmarks hand agents stronger aids than Sari's.** Original VLN (R2R, CVPR 2018,
[arXiv:1711.07280](https://arxiv.org/abs/1711.07280)) has the agent hop between panoramas on a
**pre-built navigation graph**; VLN-CE (ECCV 2020,
[arXiv:2004.02857](https://arxiv.org/abs/2004.02857)) characterizes that setting, verbatim, as
"a sparse graph of panoramas with edges corresponding to navigability" presuming "**known
environment topologies, short-range oracle navigation, and perfect agent localization**" — and
frames removing the graph as its *own research contribution* ("this setting lifts a number of
assumptions implicit in prior work"). Canonical ObjectNav (Batra et al. 2020,
[arXiv:2006.13171](https://arxiv.org/abs/2006.13171)) equips every agent with "an RGB-D camera and
a **GPS+Compass sensor**", both "idealized" — perfect localization, for free, by task definition.
Sensor suites and scaffolds are *task-definition choices* the field standardizes deliberately
(Anderson et al., [arXiv:1807.06757](https://arxiv.org/abs/1807.06757), which also standardized
SPL). Nobody calls an R2R agent or a GPS+Compass ObjectNav agent a cheat; hundreds of accepted
papers are built on exactly those aids. **Sari gives itself less than the benchmarks give**: it has
no oracle localization — it *builds* localization and topology from its own LiDAR.

**2. The legitimate line is privileged information, and Sari is on the right side of it.** The
convention running through these task definitions is that an agent may use anything its own sensors
and compute can produce, while *oracle access* (ground-truth maps handed over, privileged simulator
state, oracle stops) must be declared and separated. Everything in Sari's map is self-acquired:
LiDAR scans → occupancy grid → skeleton graph → shelf checkpoints → VLM annotations of its own
camera images, all through the same WebSocket API the agent acts through (`explore.py`,
`build_shelf_graph.py`, `capture_walk.py`). The sim offers no privileged scene query at all — the
`RequestAnnotation`/`RequestJson` stubs are unwired (`env.py`), and the 250-SKU catalog is used only
*offline* to *evaluate* annotation accuracy, not at runtime. A SLAM map is not the answer key; it is
the agent's own homework.

**3. "Fully autonomous" does not mean "end-to-end neural."** Autonomy is about no human in the loop
at deployment. Mars rovers and self-driving stacks are autonomous *because of* their mapping and
search layers, not despite them. The system that wanders into walls until a human rescues it
(`NavReasonPlan.md`'s documented behavior) is the less autonomous one. Exploration, mapping,
annotation, and retrieval in Sari all run unattended end to end — that *is* the fully autonomous
system; the VLM is one organ inside it.

**4. And the thesis remains a VLM thesis.** The VLM is load-bearing where VLMs are strong:
Stage-1 scene classification (~40% of checkpoints are bare walls by construction —
`phase3_vlm_annotation_pass.md`), open-vocabulary product annotation across 250 SKUs (~97–98%
precision when scoped; `CLAUDE.md:85-88`), goal interpretation, and close-range verification at
pickup. The measured contrast between the VLM scoped ("what is on this shelf") and unscoped (global
navigation) *is itself a thesis result* — arguably the central one.

---

## 8. What this means for the thesis (recommendations)

1. **Reframe the contribution, don't shrink it.** Not "we hard-coded navigation to help the VLM"
   but: *an empirical division-of-labor study — we measured that zero-shot VLM spatial navigation
   fails (ours and the field's numbers agree), identified which functions belong to which substrate,
   and built an LLM-consumable spatial-semantic map (checkpoint graph + per-shelf product
   annotations) that makes a VLM store agent reliable.* That is an LLM-Modulo instantiation with a
   novel artifact, in step with LM-Nav → Mobility VLA.
2. **Run the Phase-4.1 live A/B to completion — before the next adviser meeting if possible.** The
   harness is built, fair (VLM gets a superset of A*'s information), and instrumented
   (`crosses_obstacle_rate`, blocked-rate, coverage recall, tokens, latency;
   `nav_metrics.py:79-113`). `output_vlm/` currently holds a ~130-step VLM-arm run but **no finished
   report files** — the numbers the defense wants do not exist on disk yet. Report all three arms
   (`astar`, `vlm`, `vlm-goal`); §5.4 makes `vlm-goal` the constructive olive branch to the
   adviser — a VLM aid, tested fairly, where it might genuinely win.
3. **Offer the adviser the crisp experimental claim:** "If the VLM arm matches A* on coverage
   without crossing mapped obstacles, we will adopt it." That converts a philosophical dispute into
   a measurement — the strongest possible answer to "you're cheating" is "we built the instrument
   that would prove you right, and here is what it measured."
4. **Fix the misattributed citations before anyone checks them (§9).**
5. **Cite your own measured VLM-perception findings** (reading-distance/resolution, prompt A/Bs that
   made things worse and were reverted, the guided_json negative control —
   `annotator_sys_inst.py:43-98`) as methodology: the project already practices
   measure-don't-assume, which is exactly the epistemic stance this dispute needs.

---

## 9. Corrections, do-not-cite list, and caveats

**Must fix in `VMap_Plan.md` (both verified absent from the cited papers):**
- "SemExp reduces navigation steps by 34%" and "found objects 16% faster" — **not in
  [arXiv:2007.00643](https://arxiv.org/abs/2007.00643)** (its abstract states no comparative
  numbers). Real citable numbers: challenge win 25.3% vs 18.8% runner-up; Gibson SPL/SR 0.199/0.544
  vs ANS baseline 0.145/0.446.
- "ANS reduced collision rate by 62%" — **not in [arXiv:2004.05155](https://arxiv.org/abs/2004.05155)**;
  the paper reports no collision-rate comparison at all (coverage metrics only).

**Refuted in adversarial verification (0–3) — do not use:** "VLMs are competent egocentrically but
systematically fail allocentrically." ViewSpatial-Bench shows weakness in *both* frames. Argue "the
whole spatial axis is weak," not an ego/allo asymmetry.

**Phrasing traps (verified nuances):**
- NavGPT's 34%-vs-72% and NaVid's 37.4/66% figures are **paper-table/body numbers**, not abstract
  claims — cite as in-paper results.
- Valmeekam ~12% is the **cross-domain autonomous-mode average** ([arXiv:2305.15771](https://arxiv.org/abs/2305.15771));
  GPT-4 plain Blocksworld is ~34%, collapsing to 0.16–4.3% on *obfuscated* Blocksworld
  ([arXiv:2409.13373](https://arxiv.org/abs/2409.13373)). Don't write "12% on obfuscated Blocksworld."
- PoliFormer won CoRL 2024's **Outstanding Paper Award** (not "best paper").
- CoW's framing is "no additional training"/zero-shot, not "gradient-free."
- o3-with-image-tools ~90% on BlindTest is **vendor-reported, not peer-reviewed**.
- Date-qualify benchmark numbers to their model generation (BLINK/VSI-Bench/SpatialVLM/BlindTest =
  GPT-4V/Gemini-1.5 era; ViewSpatial = GPT-4o era; MV-RoboBench = GPT-5 era). The gaps narrow but
  none has closed as of July 2026.
- Flag our interpretive bridges as such when writing: "BLINK's worst tasks are SLAM primitives,"
  "specialist superiority supports module delegation," "GT-map gain ≈ what SLAM provides" — each is
  our inference atop verified numbers, defensible but not the papers' own sentences. Scope caveats:
  the VSI-Bench cognitive-map gain is one model on the relative-distance subtask (maps did not help,
  and sometimes hurt, other subtasks); MV-RoboBench is manipulation-scene reasoning, not VLN; Gervet
  et al. tested one end-to-end policy in six homes.
- NaVILA's RSS 2025 and InstructNav's CoRL 2024 venues were confirmed via proceedings/OpenReview
  pages rather than the arXiv pages themselves.

**Old-overhaul failure evidence is qualitative.** `NavReasonPlan.md:5` documents the failure mode
but records no success-rate numbers — which is precisely why recommendation §8.2 (run the A/B)
matters; don't present the old agent's failure as a measured rate.

---

## 10. Bibliography (grouped; verification status marked)

**Verification legend:** [A] = 3–0 adversarial verification (three independent refutation attempts
against the primary source, all failed); [P] = primary source fetched and quotes/venue confirmed in
this pass; [C] = canonical classical reference, DOI supplied from standard records but not re-fetched
in this pass — confirm via any library portal before final submission.

*VLM spatial-reasoning limits:*
1. [A] Yang et al., "Thinking in Space: How Multimodal Large Language Models See, Remember, and
   Recall Spaces" (VSI-Bench), **CVPR 2025**. https://arxiv.org/abs/2412.14171
2. [A] Fu et al., "BLINK: Multimodal Large Language Models Can See but Not Perceive", **ECCV 2024**.
   https://arxiv.org/abs/2404.12390
3. [A] Rahmanzadehgervi et al., "Vision language models are blind" (BlindTest), **ACCV 2024 oral**.
   https://arxiv.org/abs/2407.06581
4. [A] Chen et al., "SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning
   Capabilities", **CVPR 2024**. https://arxiv.org/abs/2401.12168
5. [A] Li et al., "ViewSpatial-Bench: Evaluating Multi-perspective Spatial Localization in
   Vision-Language Models", 2025. https://arxiv.org/abs/2505.21500
6. [A] "Seeing Across Views: Benchmarking Spatial Reasoning of Vision-Language Models in Robotic
   Scenes" (MV-RoboBench), **ICLR 2026**. https://arxiv.org/abs/2510.19400
7. [P] Ramakrishnan, Wijmans, Kraehenbuehl, Koltun, "Does Spatial Cognition Emerge in Frontier
   Models?" (SPACE), **ICLR 2025**. https://arxiv.org/abs/2410.06468

*Modular vs end-to-end:*
8. [A] Gervet, Chintala, Batra, Malik, Chaplot, "Navigating to Objects in the Real World",
   **Science Robotics 8(79), 2023**. https://arxiv.org/abs/2212.00922 ·
   https://www.science.org/doi/10.1126/scirobotics.adf6991
9. [P] Chaplot et al., "Object Goal Navigation using Goal-Oriented Semantic Exploration" (SemExp),
   **NeurIPS 2020**; winner, CVPR 2020 Habitat ObjectNav Challenge. https://arxiv.org/abs/2007.00643
10. [P] Chaplot et al., "Learning to Explore using Active Neural SLAM", **ICLR 2020**; winner, CVPR
    2019 Habitat PointGoal Challenge. https://arxiv.org/abs/2004.05155
11. [P] Gupta et al., "Cognitive Mapping and Planning for Visual Navigation", **CVPR 2017**.
    https://arxiv.org/abs/1702.03920

*LLM/VLM agents over deterministic structure:*
12. [P] Ahn et al., "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances" (SayCan),
    **CoRL 2022**. https://arxiv.org/abs/2204.01691
13. [P] Shah et al., "LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision,
    and Action", **CoRL 2022**. https://arxiv.org/abs/2207.04429
14. [P] Huang et al., "Visual Language Maps for Robot Navigation" (VLMaps), **ICRA 2023**.
    https://arxiv.org/abs/2210.05714
15. [P] Gu et al., "ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning",
    **ICRA 2024**. https://arxiv.org/abs/2309.16650
16. [P] Huang et al., "VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language
    Models", **CoRL 2023 oral**. https://arxiv.org/abs/2307.05973
17. [P] Chen et al., "Open-vocabulary Queryable Scene Representations for Real World Planning"
    (NLMap), **ICRA 2023**. https://arxiv.org/abs/2209.09874
18. [P] Rana et al., "SayPlan: Grounding Large Language Models using 3D Scene Graphs for Scalable
    Robot Task Planning", **CoRL 2023 oral**. https://arxiv.org/abs/2307.06135
19. [P] Liu et al., "OK-Robot: What Really Matters in Integrating Open-Knowledge Models for
    Robotics", **RSS 2024**. https://arxiv.org/abs/2401.12202
20. [P] Xu et al., "Mobility VLA: Multimodal Instruction Navigation with Long-Context VLMs and
    Topological Graphs", **CoRL 2024**. https://arxiv.org/abs/2407.07775
21. [P] Gadre et al., "CoWs on Pasture: Baselines and Benchmarks for Language-Driven Zero-Shot
    Object Navigation", **CVPR 2023**. https://arxiv.org/abs/2203.10421

*LLM/VLM-as-navigator and LLM planning limits:*
22. [P] Zhou, Hong, Wu, "NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large
    Language Models", **AAAI 2024**. https://arxiv.org/abs/2305.16986
23. [P] Chen et al., "MapGPT: Map-Guided Prompting with Adaptive Path Planning for Vision-and-
    Language Navigation", **ACL 2024**. https://arxiv.org/abs/2401.07314
24. [P] Long et al., "InstructNav: Zero-shot System for Generic Instruction Navigation in Unexplored
    Environment", **CoRL 2024**. https://arxiv.org/abs/2406.04882
25. [P] Valmeekam et al., "PlanBench: An Extensible Benchmark for Evaluating Large Language Models
    on Planning and Reasoning about Change", **NeurIPS 2023 D&B**. https://arxiv.org/abs/2206.10498
26. [P] Valmeekam et al., "On the Planning Abilities of Large Language Models: A Critical
    Investigation", **NeurIPS 2023 Spotlight**. https://arxiv.org/abs/2305.15771 (earlier preprint:
    https://arxiv.org/abs/2302.06706; LRM follow-up with PlanBench tables:
    https://arxiv.org/abs/2409.13373)
27. [P] Kambhampati et al., "Position: LLMs Can't Plan, But Can Help Planning in LLM-Modulo
    Frameworks", **ICML 2024**. https://arxiv.org/abs/2402.01817

*End-to-end counter-evidence:*
28. [P] Zhang et al., "NaVid: Video-based VLM Plans the Next Step for Vision-and-Language
    Navigation", **RSS 2024**. https://arxiv.org/abs/2402.15852
29. [P] Cheng et al., "NaVILA: Legged Robot Vision-Language-Action Model for Navigation",
    **RSS 2025**. https://arxiv.org/abs/2412.04453
30. [P] Zhang et al., "Uni-NaVid: A Video-based Vision-Language-Action Model for Unifying Embodied
    Navigation Tasks", **RSS 2025**. https://arxiv.org/abs/2412.06224
31. [P] Ehsani et al., "SPOC: Imitating Shortest Paths in Simulation Enables Effective Navigation
    and Manipulation in the Real World", **CVPR 2024**. https://arxiv.org/abs/2312.02976
32. [P] Zeng et al., "PoliFormer: Scaling On-Policy RL with Transformers Results in Masterful
    Navigators", **CoRL 2024 Outstanding Paper**. https://arxiv.org/abs/2406.20083

*Benchmark conventions and framing:*
33. [P] Anderson et al., "Vision-and-Language Navigation: Interpreting visually-grounded navigation
    instructions in real environments" (R2R), **CVPR 2018**. https://arxiv.org/abs/1711.07280
34. [P] Krantz et al., "Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous
    Environments" (VLN-CE), **ECCV 2020**. https://arxiv.org/abs/2004.02857
35. [P] Batra et al., "ObjectNav Revisited: On Evaluation of Embodied Agents Navigating to Objects",
    2020. https://arxiv.org/abs/2006.13171
36. [P] Anderson et al., "On Evaluation of Embodied Navigation Agents", 2018 (defines the task
    taxonomy and SPL). https://arxiv.org/abs/1807.06757
37. [P] Sutton, "The Bitter Lesson", 2019.
    http://www.incompleteideas.net/IncIdeas/BitterLesson.html

*Classical foundations of the deterministic layer (60 years of prior art):*
38. [C] Hart, Nilsson, Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost
    Paths" (A*), IEEE Trans. SSC 4(2), **1968**. DOI 10.1109/TSSC.1968.300136
39. [C] Elfes, "Using Occupancy Grids for Mobile Robot Perception and Navigation", IEEE Computer
    22(6), **1989**. DOI 10.1109/2.30720
40. [C] Yamauchi, "A Frontier-Based Approach for Autonomous Exploration", IEEE CIRA, **1997**.
    DOI 10.1109/CIRA.1997.613851
41. [C] Thrun, Burgard, Fox, *Probabilistic Robotics*, MIT Press, **2005**.

*Further reading (link not machine-verifiable in this pass — OpenReview blocks automated fetches):*
42. LeCun, "A Path Towards Autonomous Machine Intelligence", 2022 — position paper proposing a
    modular cognitive architecture (perception, world model, cost, actor as separate modules);
    search "LeCun A Path Towards Autonomous Machine Intelligence OpenReview".

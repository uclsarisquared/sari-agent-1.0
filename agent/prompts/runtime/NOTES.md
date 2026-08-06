# Runtime prompt design history

FULL v2 applied 2026-07-20 (plans/system_instructions_v2.md §1). Structure change: the 13-field
agent-state block is defined ONCE as AGENT_STATE_DOC and interpolated into both prompts (it had
diverged between the two copies), and it now carries the isColliding-is-unreliable caveat
(CLAUDE.md standing constraint). SYS_INST_ASSOCIATIVE_SEMANTIC and SYS_INST_VLM_LEAN were rewritten
to v2: STOP is in the mode enum/list (the runtime halts ONLY on mode=="STOP"); the VLM output
contract matches the parser (one ```json fence, Python-literal True/False/None for ast.literal_eval);
location vocabulary is "Checkpoint N" (memory_gen keys the rendered memory that way, never
"Shelf N"); the VLM's three contradictory orientation directives are merged into one phase-precedence
rule; rule "pan to make space" -> move_backward; move_forward is described as relative to current
heading; the 1920x1080/(960,540) hardcode is gone. The `notes` schema keys are unchanged
(orchestrators read them). Episodic is unchanged.

MEASURED - 2026-07-20 A/B (qwen Qwen3.6-27B, identical Ritz-shelf capture + state, K=5):
  * The earlier minimal subset (STOP enum, output-contract line) was measurement-neutral: the OLD
    prompt already emitted STOP 5/5 and parsed 5/5. The full v2 here was re-A/B'd for parse/validity
    no-regression before keeping.
  * NOTE the full VLM_LEAN rewrite is a BEHAVIOURAL change (merged phase rules, obstacle handling):
    parseability was A/B'd offline, but the behavioural win (e.g. fewer wrong-mode actions like the
    perception-mode move_forward seen in a Ritz run) needs a sim task-level A/B to confirm.

CENTERING/GRAB HANDSHAKE edit (2026-07-21, UNMEASURED - user to A/B): prompted by pickup run
0721_193928 where the actor blind-panned 7.5deg, declared "centred", fired extend_arm_until_grabbed
while still in *perception* (blocked - wasting a step), then closed on empty air. Two changes:
(1) the ACTOR is told to centre via `center_object_on_screen` (the closed-loop tool) before grabbing
and to HOLD rather than fire a grab early while still in perception; (2) the ROUTER switches to
*manipulation* proactively once the target is centred and within reach, instead of waiting for a
blocked grab to trip `last_action_blocked`. Behavioural - needs a sim task-level A/B (the "green chip
bag" case + the standard five) before believing it. Unrelated but stacked in that run: the null grab
was a top-shelf vertical-reach miss (hand reaches at body height), which is the separate Phase-D
crouch/hand-height item, not this handshake.

CENTRING FEEDBACK edit (2026-07-21, UNMEASURED): added state field `last_center` (AGENT_STATE_DOC o)
carrying center_object_on_screen's SUCCESS/FAILED/STALLED outcome, surfaced by both runner loops
(pickup_navigation, subtask_agents) and logged in the pickup JSONL. The tool already returned its result,
but the loops dropped everything except 'blocked', so a silent success let the episodic learner
wrongly conclude "avoid center_object_on_screen". The failure strings are actionable (bring the
target into view) so the lesson can't become "stop centring". Perception also early-stops on a
stalled residual instead of grinding out max_iters.

GRAB RECOVERY edit (2026-07-21, UNMEASURED, user directive): extend_arm_until_grabbed no longer
creeps the body forward on a short reach - that motion drifted the agent off the centred target and
it grabbed the NEIGHBOUR. It now does one reach and reports out-of-reach; the recovery (rule 5,
field l) is move-closer -> RE-CENTER (center_object_on_screen) -> retry, matching the "moving
de-centres the target" finding.

PHASE D depth-gated reach (2026-07-22, UNMEASURED - user to A/B): added state field `last_reach`
(AGENT_STATE_DOC p) carrying plan_reach's MEASURED verdict from RequestLidarCenter - move N (metric) /
crouch (too low) / too-high / re-center - set in both runner loops beside last_grab_failed. Rule 5
now prefers this measured guidance (move the MEASURED N, not "a little"); last_grab_failed stays the
fallback when there is no measurement. The reach ENVELOPE (manipulation.REACH_ENVELOPE) is a first
guess until calibrated - A/B the verdicts against actual grabs on the five items + a bottom-shelf
case before believing. v1 reported crouch only; AUTO-CROUCH added 2026-07-23 (user directive after the
live grabber eval): the grab tool resolves a `crouch` verdict itself - crouch -> re-center -> re-measure
-> grab -> ALWAYS stand (finally-guarded), so posture is never the actor's job and can never leak
(subtask_agents._crouched_grab). last_reach's CROUCH forms became "CROUCHED ..." accordingly.

MODE/ACTION HORIZON edit (2026-07-23, UNMEASURED - user to A/B on identical (frame, state) pairs):
the semantic learner routes the step's MODE one call BEFORE the actor picks its ACTION, and it does so
off a `recall` that often narrates SEVERAL steps ("release the wrong item, then center, then switch to
manipulation and grab"). Two readers of one multi-step plan then desync: the learner anchors `mode` on
one step while the actor acts on another - so the actor fires a grab in *perception* and the dispatch
mode-gate blocks it (a wasted step, recovered only the step after). Two PROMPT changes here target the
root (the CODE half is subtask_agents.run_leg's perception->manipulation grab auto-promotion): (1) the
learner now emits `next_action` - the SINGLE next step - and `mode` MUST be that action's mode, not a
later step's (learner rule 5); (2) the actor is told to execute ONLY the first not-yet-done step of
`recall` and never skip ahead to the grab while still in *perception* (actor rule 4). agent.execute_lean
also threads `next_action` into the actor prompt (soft - .get, and suppressed once the graph dispatcher
has already driven). Prompt change => A/B before believing it.

ROUTE-TRACING edit (2026-07-24, UNMEASURED - user to A/B on identical (frame, state) pairs): prompted
by orchestrator run leg 4 step 1, where the learner hand-traced a BFS through the whole connectivity
map into `recall` ("21 -> 20 -> 19 ... 54 connects to 32"), overflowed the 1536-token cap mid-string,
and killed the step (ast.literal_eval "unterminated string literal"). That path-planning is BOTH
wasted (navigation is deterministic - _graph_navigate runs A* over the store map; the learner's route
is discarded) AND the crash's root cause. Rule 2 rewritten: name the destination + at most the ONE
next checkpoint, never enumerate a chain of links. Paired CODE half in agent._parse_semantic_response
(a truncated/malformed learner reply now degrades to a navigation default instead of raising).
Prompt change => A/B (agreed-nav / parse-fail rate on identical steps) before believing it.

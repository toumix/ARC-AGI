# GoNI on ARC-AGI-1: what CLRS taught, before a line of wiring

Stage 0 of [discopy#703](https://github.com/discopy/discopy/issues/703):
the insights of the CLRS-30 study gathered in one place, each with the
evidence it rests on and what it means for ARC, plus a survey of the data
that changes the hand-off's order of work. Nothing here is trained.

The sources, read whole: [clrs#1](https://github.com/toumix/clrs/pull/1)
(`minimum`, `insertion_sort`, `lcs_length` in JAX on `discopy@main`),
[discopy#686](https://github.com/discopy/discopy/pull/686) (`kmp_matcher`
and `lcs_length` on `discopy.neural`'s map executor),
[#677](https://github.com/discopy/discopy/pull/677) (the token machine the
learning pipeline was ported from),
[#678](https://github.com/discopy/discopy/issues/678) (the CLRS hand-off),
[#683](https://github.com/discopy/discopy/issues/683) (the state ruling),
[#687](https://github.com/discopy/discopy/issues/687) (the LLM-GoNI
protocol), [#702](https://github.com/discopy/discopy/issues/702) (the one
`discopy.neural`) and #703 with its comment.

## The framework in one paragraph

A GoNI model is a diagram whose wiring is the algorithm and whose boxes
are learned. The wiring is a *family* indexed by the input size, so the
same weights run at every size; a box reads a bounded interface, never
the size, so out-of-distribution generalisation is a property of the
family rather than a hope. Execution is the geometry of interaction's
formula: `to_map` turns the diagram into a combinatorial map with one
node per box, and a run is rounds of message passing over the ports. The
boxes are supervised two ways: at their own boundary, from an oracle's
recorded traffic; or end to end, through a soft relaxation of the
routing, from the task's outputs alone.

## What CLRS taught

1. **The wiring carries the algorithm and the cell never sees `n`.**
   `minimum` 97.92 ± 1.47 at n = 64 and 96.88 at n = 256 from training
   at n = 16; `kmp_matcher` 100.0 ± 0.0 at 64 *and* 256, the pattern
   seventeen times longer than any seen in training, against 19.51 for
   the best published processor. ARC: a task's grids differ in size,
   and the survey below says the test grid has a shape no demo had in
   163 of 400 training tasks and 209 of 400 evaluation ones. The family
   is the hypothesis; the output-size rule is part of it.

2. **The interface must be discrete and quotiented over values.** The
   CLRS keys are real: 15,000 visits of `min2` collapse to two rules
   keyed by `key1 > key2` alone, and the ablation that squeezes the
   value through a scalar instead falls from 85 to 41 at n = 256 while
   the predicate box holds at 98. Every residual error of the predicate
   box is a tie band `|key1 − key2| < 0.005`. ARC colours are discrete
   already, which is not the same as being the right key: see the
   survey.

3. **Enumerate before training.** The LCS cell is eight rules and its
   equality box sixteen; `rule_table` asserts the quotient on the
   traffic and a truth-table diagnostic reads a trained cell back. At
   two to five demos a dictionary of visited rows beats an MLP wherever
   the alphabet allows it; the MLP is for the interface too wide to
   enumerate.

4. **End to end beats the oracle, and ARC only has end to end.** Soft
   routing during training (each step mixes by the predicate's own
   probability), hard routing at test, the task's output the only
   loss: 97.92 against the oracle-trained 95.83 on `minimum`, 98.97
   against 96.24 on sorting, 95.52 against 88.41 far out. ARC has no
   hints and no reference implementation, so the end-to-end rung of
   clrs#1's `endtoend.py` and #686's `train.py` is the only one.

5. **Two failure modes will come back, and both are diagnosable.**
   *Polarity*: LCS seed 0 learned `not equal`, flipped its cell rules,
   scored 100 and read as total failure until `run_lcs` learned to
   detect the convention. On ARC a learned readout absorbs any
   permutation of colours, so the readout is fixed: one-hot colours,
   no learned decoder. *Constant collapse*: LCS seed 1 settled on
   `b = 0`, worth exactly the share of matching visits, 24.66 %
   predicted and 24.66 % measured. Report per seed, never a mean of a
   bimodal run; restart and select by leave-one-demo-out.

6. **When a learned cell fails out of distribution it is value
   extrapolation, not wiring.** `kmp`'s fold carries a boolean and
   saturates; `lcs_length`'s cell carries a count and goes from 99.8 in
   distribution to 93.5 out of it. clrs#1's `lcs.py` fixes this in delta
   form, neighbouring counts differing by at most one. ARC: keep every
   wire categorical, and carry a count as a bounded difference.

7. **Symmetry is plumbing, and the map build is the cost.** The
   previous run wrote the LCS grid by hand thinking swaps made a
   diagram impossible; the grid is a symmetric diagram and `to_map`
   absorbs every permutation, leaving one node per cell. What made it
   feasible was `CMap.permutation` writing the involution directly, 94 s
   to 4 s. The n = 256 `kmp` circuit took two hours to build and a
   minute to evaluate. ARC: one map per grid shape, at most 900 cells,
   cached by shape and built from the family's combinatorics rather
   than by composing layers.

8. **Controls measure what the wiring smuggles in.** Widening the
   predicate to the baseline's 392,892 parameters gains nothing; the
   same 322-parameter box on a different correct wiring scores the
   same; a wiring folding half the keys is chance. A family that fits
   the demos is not thereby right, so a family's score is only ever
   read against its leave-one-out and the test pair.

9. **A library of primitives, shared where algorithms share
   primitives.** The sort-trained comparator drops into `minimum`'s
   wiring zero-shot and is indistinguishable from the natively trained
   one. ARC's stage 3 is this library across tasks, and re-arc's
   generators, 1,000 verified examples per training task, are the
   training distribution few-shot fitting lacks.

10. **Data-dependent control is not a static wiring.** Every task
    solved so far is data-oblivious. Loops with state, `binary_search`
    onwards, wait on the `feedback`/`stream` instance, with the state on
    a wire iterated through time and *not* the para diagonal, USER's
    ruling in #683. ARC's motion tasks, gravity and move-until-collision,
    sit there and are out of stage 1. Iteration *to a fixpoint* of a
    data-oblivious cell is different, and #585's `FixedPoint` solver
    already runs it.

11. **Verify with the benchmark's own scorer and commit the
    predictions.** `verify.py` on #686 re-scores committed predictions
    with `clrs._src.evaluation._eval_one` in a separate environment;
    `dataset.py --check` re-derives every cached answer by brute force.
    Two published numbers were also found wrong by re-reading the papers
    (`minimum`'s SOTA is 97.78, not 96.08; LCS's is 80.51, not 57.88),
    so #703's calibration figures are re-read from their papers before
    any table cites them.

12. **Two executors exist, and only one installs here.** clrs#1 runs a
    `symmetric.Functor` into `python.Function` with JAX on the wires,
    from `discopy@main`, and hand-rolls the soft routing per task; #686
    runs `MapNN`/`to_map` on torch, from #585's package, which is what
    #702 merges. Torch does not install in this sandbox (the egress
    allowlist), JAX does; clrs#1 did its runs on the Mac mini for that
    reason. Nobody has yet asked Tommaso where his CLRS work lives, as
    #678 said to before building; the same question stands for ARC.

## What the data says

`python -m goi.survey`, both splits of `data/`; every number below is a
count over 400 tasks.

| | training | evaluation |
|---|---:|---:|
| output the same size as the input | 262 | 270 |
| output of a constant size | 75 | 68 |
| output a multiple of the input | 12 | 14 |
| output smaller, varying | 45 | 45 |
| other | 6 | 3 |
| test input of a shape no demo had | 163 | 209 |
| tasks with 2 / 3 / 4 / 5+ demos | 56 / 237 / 78 / 29 | 45 / 218 / 88 / 49 |

Then the two data-oblivious families of the hand-off, recolour and
neighbourhood rule, as enumerated tables over the same-size tasks: the
cell reads a Moore neighbourhood of radius `r` in three encodings, from
raw colours to the quotients the CLRS study would suggest, and is read
off the demos as a dictionary of visited rows.

| training, cell solves the test pair, at the smallest radius passing leave-one-out | raw | relative to centre | counts |
|---|---:|---:|---:|
| recolour, `r = 0` | 4 | 4 | 4 |
| neighbourhood, `r ≤ 3` | 4 | 4 | 6 |

On the evaluation split every entry is **0**. Consistency is cheap and
transfer is not: at `r = 1` a raw table fits 55 training tasks and
passes leave-one-out on 2. The failure is never a contradiction between
demos; it is a row of the test grid that no demo visited. That is the
CLRS coverage condition, every one of the LCS cell's eight rows
exercised at n = 16, failing on an interface with billions of rows.
Quotienting the way clrs#1 did, relative to the centre and background,
halves the rows and moves the count from 4 to 6.

## What this changes in the plan

Families 1 and 2 as written cover about one per cent of the training
set and nothing of the evaluation set. The reading, marked as such: the
same-size tasks are local in their *cell* but not in one pass. What
reaches a pixel travels: a colour floods a region, a line extends until
it meets something, an object's bounding box is decided by pixels far
away. That is the neighbourhood cell iterated to a fixpoint, family 3,
and it is GoNI's execution formula as #686 runs it, rounds of message
passing over the grid's map, not a single-round table. So stage 1 starts
at family 3, with families 1 and 2 as its one-round special case, and
the cell keeps the quotiented interface of lesson 2 to keep its rows
visited.

Three more consequences, in order of leverage:

- **Write `verify.py` first**: exact match on the committed output
  grids, pass@2 as the two best candidates by leave-one-demo-out, the
  survey's counts as the first row of the results table.
- **Fit on JAX in the sandbox, run maps on the Mac mini.** A rule
  table and a soft-routing fit need nothing torch; the `MapNN`
  fixpoint runs do, until #702 lands and the pin moves.
- **Read re-arc before wiring by hand.** A verifier per training task is
  a hand wiring in a DSL, so the family of every task can be read off
  its program, and the generators turn few-shot fitting into a training
  distribution for the cell library of stage 3.

## Family 3, first numbers

USER 🚀'd the reordering on 2026-09-02, so family 3 was built first:
`fixpoint.py`, one cell per pixel reading its Moore neighbourhood
through the quotiented interface — the centre's colour, the input's, and
for every neighbour its colour, whether it equals the centre, is
background or lies past the border — and writing relatively: keep the
centre, take a neighbour's colour, or an absolute colour. The same cell
runs at every pixel for 1, 4 or 12 rounds, soft during the fit on the
demos and hard at test, with no learned readout. `run_fixpoint.py` fits
the three round counts per task, selects by leave-one-demo-out, then fit,
then fewer rounds, and writes the two best distinct predictions as the
task's attempts; `modal_run.py` runs the same thing one container per
task, which is how both splits were done in about an hour. Every result
and every prediction is committed, and `verify.py` re-scores them from
the standard library alone:

| pass@2, exact match, 400 tasks each | training | evaluation |
|---|---:|---:|
| enumerated table, families 1–2 (the survey above) | 6 | 0 |
| **learned cell, 1 / 4 / 12 rounds, family 3** (v1) | **26** | **2** |
| the same with deep supervision, a curriculum, two seeds (v2, below) | 33 | 3 |
| the same fitted on re-arc's generated examples (below) | 32 | — |
| solved by any of the three | 50 | 3 |
| of which selected at one round | 23 | 1 |
| best candidate fits every demo exactly | 60 | 23 |
| best candidate passes leave-one-out on every demo | 7 | 3 |
| best candidate passes leave-one-out on none | 215 | 249 |
| median pixel accuracy of the first attempt | 0.889 | 0.874 |

Three readings, each marked as one:

- **The learned cell beats the table four to one on the same
  interface**, 26 against 6, and 23 of those 26 are one-round: what the
  MLP adds is generalisation across rows no demo visited, the coverage
  failure the survey named, not more rounds.
- **The rounds bought almost nothing, and that is optimisation, not
  wiring.** The 4- and 12-round automata rarely fit their own demos in
  300 steps; the enclosed-region flood fill `00d62c1b` fits none of its
  five. This is lesson 5 again, the basin found or not, and the lever is
  the same as on `lcs_length`: more steps, restarts, a curriculum over
  the rounds, and a fit that reads the map's intermediate states.
- **The fit does not transfer between demos.** 215 of 262 best
  candidates reproduce none of their held-out demos, on a cell of 64
  hidden units fitted on a few thousand pixels. The table's failure was
  rows never visited; the MLP's is rows visited once. Both say the same
  thing about ARC: two to five demos are not a training distribution,
  and re-arc's generators are.

For calibration the evaluation number, 0.5 %, is the floor of a family
with no object, no count and no data-dependent control in it, from
demos alone, in a second per task. Of the figures #703 quoted for the
literature, one is verified at its source and one is not: TRM's own
README (`SamsungSAILMontreal/TinyRecursiveModels`, arXiv:2510.04871)
says *"45% on ARC-AGI-1 and 8% on ARC-AGI-2 with a tiny 7M parameters
neural network"*, trained on the training set with 1,000 augmentations;
CompressARC's README (`iliao2345/CompressARC`) gives no score, only
*"up to 20 minutes to run, on one NVIDIA GeForce RTX 4070 GPU"* per
task, and its paper and blog post are unreachable from this sandbox, so
the "~34 %" of #703 stays unverified and is not in the table. The "DSL
search ~40 %" has no source named and is dropped.

## Second round: the two levers pulled

Both readings above were tested the same evening, each as its own run
with every result committed under `results/<run>/` and re-scored by
`verify.py`.

**v2, the optimisation lever** (`results/v2/`): the target is a
fixpoint, so every round of the second half is supervised rather than
the last alone; the 4- and 12-round cells start from the cell fitted at
fewer rounds, a curriculum rather than a restart; 600 steps and two
seeds per task. It does what it was meant to on the fit side: the best
candidate fits every demo on 93 training tasks against 60 and on 48
evaluation tasks against 23, and a 4- or 12-round automaton is selected
on 96 and 87 tasks against 36 and 26. **33 of 400 training tasks solved
against 26, 3 of 400 evaluation against 2** — 25 of the 26 kept, 8 new,
13 of the 33 at 4 or 12 rounds where the first run had 3, and two of
the 27 line-drawing tasks now among them. Transfer moved little:
leave-one-out still fails on 205 of 262 and 249 of 270.

**re-arc, the data lever** (`results/v2-rearc/`, `rearc.py`): the same
cell fitted on 256 of re-arc's generated examples per task, minibatched
with the curriculum, 32 more selecting the round count, the task's own
demos never trained on and its test pair the score. 261 of the 262
same-size tasks ran; one has no same-size generated example at all (a
transpose the generator draws on rectangles). **32 of 400 solved against
26 few-shot**, and only 15 in common: seventeen tasks are solved only
with the training distribution, eleven only from the demos, so the
generated distribution is wider than the benchmark's own. The number
that matters is the ceiling it exposes: with 256 examples the best
candidate reproduces none of its 32 validation examples on 227 of 261
tasks, and where it reproduces all 32 it solves the test pair every
time, 16 of 16. **Expressible by this family is 16 to 32 tasks of 262**;
the rest is the wiring. Data does let the deep automata train, 15 of
the 32 at 4 or 12 rounds against 3 of 26 few-shot.

**What the verifiers say the wiring lacks** (`families.py`,
`rearc_primitives.json`): re-arc carries a DSL program per training
task, and the functions it calls label the task's family better than a
reading of the demos. The few-shot run's 26 solved tasks call a median
of 5 primitives against 13 for the corpus; of the same-size tasks whose
verifier draws a line (`connect`, `shoot`), 0 of 27 are solved; of those
using a mirror or a rotation, 1 of 37; of those using `objects`, 14 of
163. Lines and symmetries are not local at any radius, and objects are
local only up to their diameter. That is the next family, named: a wire
that carries something along a row or a column, and one that folds a
component.

## Third round: the review, and the scale

USER, 2026-09-02: *"review the previous attempt at ARC-AGI and scale it
further"*. The review first, read against the committed results rather
than the prose; every count below is one `python` expression over
`results/`, as of #1's head `a386759` at 21:20 UTC.

**What holds.** The rays and mirrors this round adds were named by the
previous round's own reading of the verifiers, `families.py`, so the
wiring lesson stands as written. `verify.py` re-scores every committed
prediction file to the number the README quotes, 26 / 2 / 3 / 32. The
re-arc ceiling reads as claimed: the 16 tasks whose best candidate
reproduces all 32 validation examples are solved 16 of 16, and the 17
solved only with the training distribution against 11 only from the
demos are exact.

**What does not.**

- *The training column of v2 landed at 21:30, while this was being
  written*: 124 of 262 tasks had run when the review started, and the
  first draft of this section called the column unfinished. It is not:
  33 of 400, `predictions/v2-training.json`, merged in below. What
  stands is the cost — a training split of one CPU container per task
  took the evening.
- *One number in the re-arc paragraph is wrong*: "reproduces none of its
  32 validation examples on 227 of 261 tasks" — the results say **192**
  of 261 (245 reproduce fewer than all 32, and 214 reproduce none of the
  task's own demos, which may be the count that was meant). The
  conclusion survives, the sentence did not.
- *The selection is a coin toss on most tasks*: on `v1/training`, the top
  candidate is tied with another on `(held_out, fit)` on **167 of 262**
  tasks, and the tie is broken by fewer rounds then by seed order. So
  for two thirds of the split the two attempts were the one-round cell of
  seed 0 and whatever came next, whatever the automaton did on the
  held-out pixels. The pass@2 of the first two rounds is a lower bound
  on the family by that much.
- *The ensemble had no data lever on the evaluation split*: re-arc has no
  generators there, and the round said "two to five demos are not a
  training distribution" without trying the distribution every ARC entry
  since 2020 has used, the symmetries of the square. A task rotated is
  the same task.
- *`modal_run.py` ran one container per task for ten minutes on two CPUs*,
  four hundred steps a minute. The cell is nine thousand parameters on a
  30 × 30 canvas that every task shares; nothing in it needs its own
  process, and a GPU fits a thousand canvases at once.
- Smaller: `modal_run.py` records no `seconds`, so only v1's local run
  has timings; `features` sends the cell `colours[..., :1]` twice, once
  inside the one-hot and once as "is background" — harmless; the fold of
  a two-demo task trains on one demo, which is what leave-one-out means
  there, but it is why `held_out` is at most 2 on 56 training tasks.

**What this round does about it**, each a file:

- `fixpoint.py` gains two kinds of port beside the eight neighbours,
  `ports = 'all'`: four *mirrors*, the pixel across the grid's horizontal,
  vertical, central and diagonal axis (the last only on a square grid),
  and four *rays*, the first non-background colour seen along the row or
  column in each direction, or none — a `lax.scan` along the axis, so a
  line reaches its end in one round rather than thirty. Same cell, same
  quotiented reads per port (colour, past-the-border, equal to the
  centre, background), same relative writes; the interface goes from 124
  to 228 features and 19 to 27 codes.
- `dihedral.py`: the eight symmetries of the square as relabellings of a
  task, and a vote. `batched.py`'s `pooled` kind fits the eight copies of
  the demos as one training set, holds out every copy of a demo at once,
  and undoes the eight predictions of the test input to vote. Sound
  exactly when the rule is symmetric — a gravity that falls down is not
  — which leave-one-out sees, and the `plain` kind is still there for it.
- `batched.py`: every cell of a split — task, kind, seed, fold — padded
  onto the one canvas and fitted as one `vmap` over `fixpoint.update`,
  in chunks of about a thousand canvases on one GPU; `modal_batched.py`
  maps the chunks and writes each task to a volume as its chunks come
  back, from an orchestrator that outlives the sandbox that launched it.
  Selection reads one more thing: the held-out demos' pixel accuracy
  breaks the ties above, before fit and fewer rounds.

### The numbers

Every run under `results/<run>/`, re-scored by `verify.py`; `v3` is the
cell with all sixteen ports, `plain` and `pooled` kinds, two seeds for
the plain; `v3-moore` is the first round's eight-port cell in its v2
configuration on the same batched pipeline, the control that says what
the pipeline changed on its own (nothing, within seed noise: 28 against
v2's 31 on the same 236 tasks).

| pass@2, exact match, 400 tasks each | training | evaluation |
|---|---:|---:|
| v1, family 3 few-shot (#1) | 26 | 2 |
| v2, deep supervision, curriculum, two seeds (#1) | 33 | 3 |
| v2-rearc, fitted on the generated examples (#1) | 32 | — |
| v3-moore, the v2 cell on the batched pipeline | 28 of 236 | *running* |
| **v3, sixteen ports, plain + pooled** | **56** | *running* |
| solved by any run | 76 | |

On the training split, read as of 00:30 UTC with the evaluation runs
still going:

- **56 against 33**: 30 tasks new to v2, 7 lost, and 26 that no
  previous run — v1, v2 or re-arc's training distribution — had solved.
  The verifier labels say where they came from: symmetry tasks 1 → 11
  of 37, line tasks 2 → 5 of 27, fill and recolour without objects
  10 → 20 of 63, objects 19 → 22 of 163. The mirror ports did what they
  were wired for; the rays less so, five of 27 lines.
- **The pooling carries as much as the ports.** The best held-out score
  of a task is the pooled cell's on 131 tasks, the plain cell's on 100,
  a tie on 31; of the 56 solved, 24 were selected from a pooled cell
  (18 the identity copy, 6 the vote of eight). A task rotated is the
  same task, and eight copies of three demos are a training set where
  three were not.
- **Leave-one-out became a signal.** The top candidate reproduces every
  held-out demo on 36 tasks and 30 of those are solved; it reproduces
  none on 168 (215 in v1), and 7 of those are solved anyway. The held-out
  pixel accuracy breaks ties on all but 105 tasks, most of them with
  nothing to choose between.
- **Rounds are still one**: 40 of the 56 at one round, 12 at four, 4 at
  twelve. A ray is a round of thirty in one, which is part of why.

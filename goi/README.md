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
| **learned cell, 1 / 4 / 12 rounds, family 3** | **26** | **2** |
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

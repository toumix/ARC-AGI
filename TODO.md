> gather all the insights we got from CLRS to apply the GoNI framework to ARC-AGI

> read this issue as context https://github.com/discopy/discopy/issues/703

> 🚀 (USER, 2026-09-02: stage 1 starts at family 3, families 1–2 as its one-round case)

- [x] gather the CLRS insights with their evidence and ARC consequence, `goi/README.md`
- [x] survey the data for the hand-off's families, `goi/survey.py`, every number one command
- [x] `goi/verify.py`: exact match on committed output grids, pass@2, before any family
- [x] family 3 first: the neighbourhood cell iterated to a fixpoint on the grid map, families 1–2 its one-round case — 26/400 training, 2/400 evaluation
- [x] `goi/modal_run.py`: the sweep one container per task on Modal
- [x] the deep rounds do not fit: deep supervision, a curriculum over the rounds, two seeds, 600 steps (v2) — 33/400 training, 3/400 evaluation; fits more, transfers little better
- [x] the fit does not transfer between demos: re-arc's generators as the training distribution — 32/400, and the family's ceiling measured at 16–32 of 262
- [ ] a cell library shared across tasks, trained on re-arc
- [ ] the next family: a wire along rows and columns (lines, symmetries), a fold over a component (objects)
- [x] re-read #703's calibration figures from their sources: TRM verified (45 % / 8 %, 7M), CompressARC unreachable here, DSL search unsourced
- [x] read re-arc's verifiers to label the families: `families.py`, lines 0/27, symmetry 1/37, objects 14/163 solved

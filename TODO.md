> gather all the insights we got from CLRS to apply the GoNI framework to ARC-AGI

> read this issue as context https://github.com/discopy/discopy/issues/703

> 🚀 (USER, 2026-09-02: stage 1 starts at family 3, families 1–2 as its one-round case)

- [x] gather the CLRS insights with their evidence and ARC consequence, `goi/README.md`
- [x] survey the data for the hand-off's families, `goi/survey.py`, every number one command
- [x] `goi/verify.py`: exact match on committed output grids, pass@2, before any family
- [x] family 3 first: the neighbourhood cell iterated to a fixpoint on the grid map, families 1–2 its one-round case — 26/400 training, 2/400 evaluation
- [x] `goi/modal_run.py`: the sweep one container per task on Modal
- [ ] the deep rounds do not fit: more steps, restarts, a curriculum over the rounds, supervision of the map's intermediate states
- [ ] the fit does not transfer between demos: re-arc's generators as the training distribution, a cell library shared across tasks
- [ ] re-read #703's calibration figures (CompressARC, TRM, DSL search) from their papers before any table cites them
- [ ] read re-arc's verifiers to label the families rather than the demos by hand

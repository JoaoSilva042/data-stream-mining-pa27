# PA27 tests

These tests validate the expected behaviour of the streaming pipeline from the assignment specification, rather than reproducing the implementation internally.

They cover:

- data acquisition helpers and payload validation;
- pooled warm-up threshold and target construction;
- sliding-window behaviour with hand-calculated examples;
- leakage-free `t-1` and rolling features;
- bounded feature memory;
- prequential call order (`build -> predict -> evaluate -> learn -> update`);
- warm-up/evaluation state continuity;
- confusion-matrix metrics and computational summaries;
- the real Porto/Braganca datasets (timeline, threshold, period sizes, quality, and first 2018 temporal features);
- model factories and the 20-experiment runner when River is available.

Run from the project root:

```bash
uv run python -m unittest discover -s tests -v
```

No extra test dependency such as `pytest` is required.

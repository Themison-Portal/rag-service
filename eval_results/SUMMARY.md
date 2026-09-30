# Eval Results — UC Protocol Suite

## Arm 1: GENERATION_PROVIDER=anthropic, CONTEXTUAL_PROVIDER=anthropic
- Pass rate: 65.38% (51/78 passed, 27 failed)
- Runtime: 1645.85s (~27 min)
- Key failures:
  - 1 unanswerable question incorrectly answered (likely dataset mislabel — MOA question)
  - 2/4 follow-up context cases failed (condensation not resolving anaphora)
  - 1/4 ambiguous cases failed (silently assumed interpretation instead of asking)
  - Minor contextual-precision ranking misses on 2 factual cases

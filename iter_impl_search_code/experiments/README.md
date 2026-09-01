# Experiment bundles

Each implemented algorithm has one directory containing its writeup, exact
algorithm construction, and `results.csv`. New runs store one result row per
training step. Runs completed before per-step instrumentation was introduced
retain their aggregate writeup and explicitly mark unavailable step data rather
than being rerun solely to backfill telemetry.

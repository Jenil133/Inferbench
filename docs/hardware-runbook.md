# InferBench hardware runbook

## Automated preflight

Run `inferbench preflight --config FILE`; hardware `run` also enforces it. The machine-readable result is saved in the immutable manifest. It checks server/tool resolution, configured model artifacts, visible GPU count versus maximum TP, taskset availability and CPU-list syntax, output writability, all resolved commands, and required/best-effort Nsight policy.

Preflight cannot establish model-engine/runtime compatibility, model quality, GPU health, thermal stability, actual NUMA locality, or the streaming API contract before server startup. Confirm separately:

1. Engine artifacts match the installed TensorRT-LLM/CUDA stack and have independent quality approval.
2. `nvidia-smi topo -m` and `lscpu -e` support the chosen GPU/CPU placement.
3. Clocks, power, thermals, CPU governor, background load, and networking are controlled.
4. The endpoint returns standards-framed SSE with generated text, final non-negative integer `usage.completion_tokens`, finish reason, and `[DONE]`.
5. The configured output parent has enough space for logs and profiles.

## Execution design

Concurrency and sample count are independent. Choose `request_count` large enough for tail estimates; p99 with very small samples is not publication-grade. `independent_repetitions` restarts the server and captures carryover/thermal variation that trials within one process cannot. The default alternates pinned/unpinned order across repetitions; randomized order is seeded. Archive the manifest’s exact execution plan.

A point starts a fresh process group, runs excluded warm-ups, executes measured trials under request and trial deadlines, and performs bounded TERM/KILL cleanup. Request errors remain data rather than erasing successful sibling requests. An all-failed warm-up stops the point. Failed runs preserve atomic point checkpoints and may be explicitly resumed only with a matching configuration and plan.

At high concurrency, independently verify that the colocated Python client and network are not bottlenecks. InferBench does not measure client saturation.

## CPU affinity

`taskset` pins the serving command, not the Python load generator. Select GPU-local server CPUs without housekeeping/load-generator overlap. Pinning can improve or worsen results. Compare all independent repetitions and trial spread; never promote the best run.

## Nsight scope and terminology

The extracted metric is **selected OS-runtime wait/blocking time**. It is not a direct scheduler-delay measurement and is not called a host stall. Blocking APIs can indicate normal synchronization.

Current Nsight capture spans the complete server process lifetime. The client is outside the profiled process tree, and InferBench does not inject valid server-side trial markers. Consequently, totals include startup, warm-up, measured trials, and shutdown and cannot be apportioned to trials. Reports make only an exploratory whole-profile association and no causal claim. For temporal analysis, instrument the serving process with NVTX or inspect the timeline using server/request identifiers.

With `nsight.required=false`, absent `nsys`, missing artifacts, and extraction failures are explicit nonfatal statuses. With `required=true`, preflight or extraction failure blocks completion.

## Interpretation and archive

- Pooled p50/p99 use successful requests and state sample counts; failures have separate counts and records.
- Trial summaries retain medians and IQR. Throughput is successful exact completion tokens divided by whole trial wall time.
- JSON and manifest are authoritative. CSV names explicitly identify success-only metrics.
- Synthetic outputs are visibly labeled and must never be cited as hardware measurements.
- Archive config source, manifest, schemas, JSON/CSV/Markdown, point checkpoints, logs, profiles, image digest, model artifact identity, topology, clocks/power, and external quality evidence. Keep secrets in environment variables.

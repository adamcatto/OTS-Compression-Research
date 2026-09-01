# Iterative Implementation Search

In this repo, we house code to run agentic iterative implementation search over the space of online compression algorithms over ragged tensors.

It is inspired by evolutionary algorithms / agent-driven autoresearch. We start with a seed implementation, then agents iterate on modification to the algorithm / approach.

Each completed iteration is bundled under `experiments/<ID>/` with its
writeup, exact algorithm construction, and per-step result table. See
`ITERATION_PROTOCOL.md` for the required schema and legacy-data policy.


# Adaptive Symbiotic Information-Sharing (ASIS)

**"Adaptive Symbiotic Information-Sharing Framework Using Large Language Models for Heterogeneous Multi-Robot Collaboration"**

The project studies AGV--picker coordination in a TA-RWARE-style warehouse environment, where successful task execution requires coordinated loading, delivery, and unloading between different types of agents.



## Project Overview

The warehouse task involves heterogeneous agents with different roles:

The warehouse task involves heterogeneous agents and task-related locations:

- **AGVs** transport shelves or racks between storage cells and goal locations.
- **Pickers** support AGVs during cooperative loading and unloading.
- **Requested racks** are the racks currently required by the warehouse request queue.
- **Goal locations** are delivery points where requested racks are completed.
- **Empty rack locations** are available storage cells where delivered racks can be returned after delivery.

The main research focus is whether allowing picker-side evaluation to influence commitment formation can improve coordination outcomes compared with a non-mutualistic architecture where the AGV side commits first and the picker side only responds afterwards.

## Coordination Methods

### Mutualistic LLM Planner

The mutualistic planner uses a staged LLM-based coordination framework in which picker-side evaluation can influence the final commitment before execution.

In this framework, the AGV side proposes candidate targets, the picker side evaluates supportability, and the final commitment can be shaped by support-side information.

Main planner file:

```text
src/symco/planners/mutualistic_llm_planner.py
```

Main runner:

```text
scripts/run_mutualistic_llm.py
```

### Non-Mutualistic LLM Planner

The non-mutualistic planner is the main staged LLM baseline.

In this baseline, the AGV side first commits to a rack target. The picker side then evaluates only that already committed target and returns an ACK/BUSY response.

The staged structure is:

1. **Stage 1:** AGV-side unilateral rack commitment.
2. **Stage 2:** Picker-side ACK/BUSY response for the committed rack.
3. **Stage 3:** Deterministic final assignment integration.

Main planner file:

```text
src/symco/planners/non_mutualistic_llm_planner.py
```

Main runner:

```text
scripts/run_non_mutualistic_llm.py
```

### Heuristic Baseline

The heuristic policy is used as a rule-based warehouse reference.

It provides a domain-grounded comparison point for overall throughput, especially completed deliveries. It is not intended to represent the same LLM-based process-level coordination mechanism as the mutualistic and non-mutualistic planners.

## Repository Structure

```text
.
├── scripts/
│   ├── run_mutualistic_llm.py
│   ├── run_non_mutualistic_llm.py
│   └── ...
├── src/
│   └── symco/
│       ├── core/
│       │   └── state_builder.py
│       ├── llm/
│       │   └── ...
│       ├── planners/
│       │   ├── mutualistic_llm_planner.py
│       │   ├── non_mutualistic_llm_planner.py
│       │   └── ...
│       └── run/
│           └── runner.py
├── task-assignment-robotic-warehouse/
├── outputs/
└── README.md
```

## Installation

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The experiments require the TA-RWARE environment and the project dependencies used by the planners, runners, and LLM clients.

## LLM Backend

The LLM-based planners require an LLM backend compatible with the planner client configuration.

Relevant code is located under:

```text
src/symco/llm/
```

Before running LLM experiments, make sure the model server and related configuration are available.

## Running Experiments

### Run the Mutualistic LLM Planner

Example command:

```bash
python3 scripts/run_mutualistic_llm.py \
  --env_id tarware-medium-6agvs-6pickers-partialobs-v1 \
  --num_seeds 4 \
  --repeats_per_seed 3 \
  --seed 0 \
  --max_steps 1000 \
  --out_dir outputs/mutualistic_6_6
```

### Run the Non-Mutualistic LLM Planner

Example command:

```bash
python3 scripts/run_non_mutualistic_llm.py \
  --env_id tarware-medium-6agvs-6pickers-partialobs-v1 \
  --num_seeds 4 \
  --repeats_per_seed 3 \
  --seed 0 \
  --max_steps 1000 \
  --out_dir outputs/non_mutualistic_6_6
```

### Run the Two Main LLM Pipelines

```bash
bash scripts/run_two_pipelines.sh
```

## Experimental Settings

The main experiments evaluate three AGV--picker ratios:

```text
3:6
6:6
6:3
```

Example environment IDs:

```text
tarware-medium-3agvs-6pickers-partialobs-v1
tarware-medium-6agvs-6pickers-partialobs-v1
tarware-medium-6agvs-3pickers-partialobs-v1
```

These settings are used to examine how coordination performance changes when picker support is abundant, balanced, or scarce.

## Output Files

Experiment outputs are written to the selected output directory, usually under:

```text
outputs/
```

Typical output files include:

```text
*_summary.json
*.jsonl
```

The summary file contains episode-level metrics.

The JSONL file contains step-level records, including agent states, actions, communication outputs, final assignments, and debug information.

## Main Evaluation Metrics

### Completed Deliveries

The number of successfully completed rack deliveries in an episode.

### Average Execution Time over All Assignments

The average duration of cooperative assignments, including both completed and incomplete assignments.

This metric captures the execution burden imposed by cooperative assignments, not only the assignments that finish successfully.

### Triggered Coordination Steps

The number of steps in which event-based coordination is triggered.

### Assignment Count

The number of cooperative assignments formed by the planner.

### Assignment Success Rate

The proportion of cooperative assignments that lead to successful cooperative execution.

## Event-Based Coordination Triggering Mechanism

The LLM-based planners use an event-based coordination triggering mechanism.

Instead of calling the LLM at every environment step, communication is triggered when coordination is needed, such as when an eligible AGV requires cooperative support and picker resources are available.

This mechanism is used to reduce unnecessary communication while still enabling coordination when task execution requires AGV--picker cooperation.


## Reproducibility Notes

Experiments involving LLM calls may not be perfectly deterministic, even with fixed environment seeds.

Results may depend on:

- the LLM backend,
- model version,
- decoding settings,
- prompt configuration,
- random seed,
- number of repeats per seed,
- and local experiment configuration.


## License

Add license information here.

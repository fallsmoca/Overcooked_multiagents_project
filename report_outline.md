# Overcooked Multi-Agent Reinforcement Learning Report Outline

## 1. Introduction

- Task: train two cooperative agents to complete soup delivery orders in
  Overcooked within a fixed horizon.
- Motivation: Overcooked requires coordination, role allocation, sparse final
  rewards, and useful intermediate behavior.
- Codebase: PantheonRL with the Overcooked AI environment.

## 2. Environment

- Observation: featurized Overcooked game state from `overcooked_ai_py`.
- Action space: 6 discrete actions: up, down, left, right, stay, interact.
- Layouts: start with `simple`; compare on `random0`, `random1`, and optionally
  `random3` or `unident_s`.
- Rewards:
  - Sparse reward: completed soup delivery.
  - Shaped reward: onion placement, dish pickup, soup pickup, distance rewards.

## 3. Baseline Algorithms

- PPO-PPO decentralized training using PantheonRL.
- Optional ADAP/ADAP_MULT baseline if time allows.
- Explain model architecture, training timesteps, seeds, and TensorBoard logs.

## 4. Proposed Improvement

Reward shaping comparison:

- `sparse`: all intermediate shaping rewards are zero.
- `default`: original PantheonRL shaping.
- `dense`: larger intermediate rewards and distance terms.
- `balanced`: smaller task rewards with mild distance guidance.

Hypothesis: moderate shaping improves early learning and coordination, while
overly dense shaping may improve short-term behavior but reduce final delivery
optimization.

## 5. Experiments

Recommended matrix:

| ID | Layout | Agents | Reward profile | Seeds | Timesteps |
| --- | --- | --- | --- | --- | --- |
| E1 | simple | PPO-PPO | default | 0,1,2 | 50k |
| E2 | random0 | PPO-PPO | default | 0,1,2 | 100k |
| E3 | random0 | PPO-PPO | sparse | 0,1,2 | 100k |
| E4 | random0 | PPO-PPO | dense | 0,1,2 | 100k |
| E5 | random1 | PPO-PPO | default | 0,1,2 | 100k |

Metrics:

- Mean sparse return.
- Mean total return.
- Standard deviation over evaluation episodes and seeds.
- Learning curves from TensorBoard.
- Qualitative behavior from demo or recorded run.

## 6. Results

- Include table of final evaluation metrics.
- Include training/evaluation plots.
- Compare sparse/default/dense shaping.
- Discuss whether agents learned role specialization or got stuck.

## 7. Discussion

- What worked and why.
- Failure cases: collisions, both agents chasing same object, waiting near pot,
  poor role allocation.
- Limitations: small number of seeds, limited training budget, legacy Gym stack.

## 8. Conclusion

- Summarize best-performing setup.
- State whether reward shaping improved learning and coordination.
- Future work: curriculum learning, partner diversity, centralized critic, HARL.

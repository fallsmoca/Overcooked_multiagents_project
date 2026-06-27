#!/usr/bin/env python
"""Run a saved Overcooked policy pair and print the game state."""

import argparse
import sys
from pathlib import Path
from time import sleep

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer import generate_env, gen_fixed
from types import SimpleNamespace

from run_overcooked_experiment import REWARD_PROFILES


def print_state(env):
    base_env = env.unwrapped.base_env
    mdp = env.unwrapped.mdp
    print(mdp.state_string(base_env.state))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default="simple")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="default")
    parser.add_argument("--ego-load", required=True)
    parser.add_argument("--alt-load", required=True)
    parser.add_argument("--ego", default="PPO")
    parser.add_argument("--alt", default="PPO")
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    env_args = SimpleNamespace(
        env="OvercookedMultiEnv-v0",
        env_config={
            "layout_name": args.layout,
            "reward_shaping": REWARD_PROFILES[args.reward_profile],
            "horizon": args.horizon,
        },
        framestack=1,
        record=None,
    )
    env, altenv = generate_env(env_args)
    ego = gen_fixed({}, args.ego, args.ego_load)
    alt = gen_fixed({}, args.alt, args.alt_load)
    env.add_partner_agent(alt)
    env.set_ego_extractor(lambda obs: obs)

    obs = env.reset()
    total_reward = 0.0
    total_sparse = 0.0
    print_state(env)
    for step in range(args.max_steps):
        action = ego.get_action(obs, False)
        obs, reward, done, info = env.step(action)
        total_reward += float(reward)
        total_sparse += float(info.get("sparse_reward", 0.0))
        print(
            f"step={step + 1} action={action} reward={reward:.2f} "
            f"sparse={info.get('sparse_reward', 0.0):.2f}"
        )
        print_state(env)
        if done:
            break
        if args.sleep:
            sleep(args.sleep)

    print(f"total_reward={total_reward:.2f} total_sparse={total_sparse:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Train and evaluate PantheonRL agents on Overcooked layouts.

This script is intentionally narrow: it gives the project a reproducible
PPO/ADAP experiment entry point with configurable reward shaping.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer import (
    ADAP_TYPES,
    generate_env,
    generate_ego,
    generate_partners,
    gen_fixed,
    input_check,
    latent_check,
)
from overcookedgym.overcooked_utils import LAYOUT_LIST


REWARD_PROFILES = {
    "default": {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    },
    "sparse": {
        "PLACEMENT_IN_POT_REW": 0,
        "DISH_PICKUP_REWARD": 0,
        "SOUP_PICKUP_REWARD": 0,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    },
    "dense": {
        "PLACEMENT_IN_POT_REW": 5,
        "DISH_PICKUP_REWARD": 4,
        "SOUP_PICKUP_REWARD": 8,
        "DISH_DISP_DISTANCE_REW": 1,
        "POT_DISTANCE_REW": 1,
        "SOUP_DISTANCE_REW": 1,
    },
    "balanced": {
        "PLACEMENT_IN_POT_REW": 2,
        "DISH_PICKUP_REWARD": 2,
        "SOUP_PICKUP_REWARD": 4,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0.5,
        "SOUP_DISTANCE_REW": 0.5,
    },
}


def parse_json_dict(value):
    if value is None:
        return {}
    if not value.lstrip().startswith("{"):
        with Path(value).open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    return json.loads(value)


def experiment_name(args):
    return (
        f"{args.layout}_{args.ego.lower()}-{args.alt.lower()}_"
        f"{args.reward_profile}_seed{args.seed}"
    )


def make_train_args(args, run_dir):
    env_config = {
        "layout_name": args.layout,
        "reward_shaping": REWARD_PROFILES[args.reward_profile],
        "horizon": args.horizon,
    }
    ego_config = parse_json_dict(args.ego_config)
    alt_config = parse_json_dict(args.alt_config)
    for config in (ego_config, alt_config):
        if args.n_steps is not None and "n_steps" not in config:
            config["n_steps"] = args.n_steps
        if args.batch_size is not None and "batch_size" not in config:
            config["batch_size"] = args.batch_size
    return SimpleNamespace(
        env="OvercookedMultiEnv-v0",
        ego=args.ego,
        alt=[args.alt],
        total_timesteps=args.timesteps,
        device=args.device,
        seed=args.seed,
        ego_config=ego_config,
        alt_config=[alt_config],
        env_config=env_config,
        framestack=args.framestack,
        record=None,
        ego_save=str(run_dir / "models" / "ego"),
        alt_save=str(run_dir / "models" / "alt"),
        share_latent=args.share_latent,
        tensorboard_log=str(run_dir / "tensorboard"),
        tensorboard_name=experiment_name(args),
        verbose_partner=args.verbose_partner,
        preset=None,
    )


def train(args, run_dir):
    train_args = make_train_args(args, run_dir)
    input_check(train_args)
    if train_args.share_latent:
        latent_check(train_args)

    env, altenv = generate_env(train_args)
    ego = generate_ego(env, train_args)
    partners = generate_partners(altenv, env, ego, train_args)

    learn_config = {"total_timesteps": train_args.total_timesteps}
    if train_args.tensorboard_log:
        learn_config["tb_log_name"] = train_args.tensorboard_name
    ego.learn(**learn_config)

    ego.save(train_args.ego_save)
    if len(partners) == 1 and hasattr(partners[0], "model"):
        partners[0].model.save(train_args.alt_save)

    return train_args.ego_save, train_args.alt_save


def evaluate(args, run_dir, ego_path, alt_path):
    env_args = SimpleNamespace(
        env="OvercookedMultiEnv-v0",
        env_config={
            "layout_name": args.layout,
            "reward_shaping": REWARD_PROFILES[args.reward_profile],
            "horizon": args.horizon,
        },
        framestack=args.framestack,
        record=None,
    )
    env, altenv = generate_env(env_args)
    ego = gen_fixed({}, args.ego, ego_path)
    alt = gen_fixed({}, args.alt, alt_path)
    env.add_partner_agent(alt)
    env.set_ego_extractor(lambda obs: obs)

    episode_rows = []
    for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        shaped_return = 0.0
        sparse_return = 0.0
        env_return = 0.0
        steps = 0
        while not done:
            action = ego.get_action(obs, False)
            obs, reward, done, info = env.step(action)
            env_return += float(reward)
            shaped_return += float(info.get("shaped_reward", 0.0))
            sparse_return += float(info.get("sparse_reward", 0.0))
            steps += 1

        episode_rows.append({
            "episode": episode,
            "return": env_return,
            "sparse_return": sparse_return,
            "shaped_return": shaped_return,
            "steps": steps,
        })

    env.close()
    returns = np.array([row["return"] for row in episode_rows], dtype=float)
    sparse_returns = np.array(
        [row["sparse_return"] for row in episode_rows], dtype=float
    )
    summary = {
        "experiment": experiment_name(args),
        "layout": args.layout,
        "ego": args.ego,
        "alt": args.alt,
        "reward_profile": args.reward_profile,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_sparse_return": float(np.mean(sparse_returns)),
        "std_sparse_return": float(np.std(sparse_returns)),
        "episodes": episode_rows,
    }

    result_dir = run_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    json_path = result_dir / "eval_summary.json"
    csv_path = result_dir / "eval_episodes.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)

    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=LAYOUT_LIST, default="simple")
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="default")
    parser.add_argument("--ego", choices=["PPO", "ADAP", "ADAP_MULT"], default="PPO")
    parser.add_argument("--alt", choices=["PPO", "ADAP", "ADAP_MULT"], default="PPO")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--framestack", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ego-config")
    parser.add_argument("--alt-config")
    parser.add_argument("--share-latent", action="store_true")
    parser.add_argument("--verbose-partner", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ego-load")
    parser.add_argument("--alt-load")
    args = parser.parse_args()

    if args.share_latent and not (args.ego in ADAP_TYPES and args.alt in ADAP_TYPES):
        raise ValueError("--share-latent requires ADAP or ADAP_MULT agents")

    run_dir = Path(args.output_dir) / experiment_name(args)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)

    metadata = {
        "args": vars(args),
        "reward_shaping": REWARD_PROFILES[args.reward_profile],
    }
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if args.eval_only:
        if not args.ego_load or not args.alt_load:
            raise ValueError("--eval-only requires --ego-load and --alt-load")
        ego_path, alt_path = args.ego_load, args.alt_load
    else:
        ego_path, alt_path = train(args, run_dir)

    evaluate(args, run_dir, ego_path, alt_path)


if __name__ == "__main__":
    main()

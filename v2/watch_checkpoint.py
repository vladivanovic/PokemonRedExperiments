from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from red_gym_env_v2 import RedGymEnv


def latest_checkpoint(sess_path: Path) -> Path | None:
    zips = list((sess_path / "checkpoints").glob("*.zip"))
    zips += list(sess_path.glob("poke_final.zip"))
    return max(zips, key=lambda p: p.stat().st_mtime) if zips else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sess-id", default="runs")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--rom", default="../PokemonRed.gb")
    p.add_argument("--init-state", default="../init.state")
    p.add_argument("--speed", type=int, default=3,
                   help="Emulation speed multiplier; 0 = unbounded")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--reload-every", type=int, default=0,
                   help="Reload the newest checkpoint every N steps (0 = never)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    sess_path = Path(args.sess_id)
    ckpt = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(sess_path)
    if ckpt is None or not ckpt.exists():
        raise SystemExit(f"no checkpoint found under {sess_path}")
    print(f"loading {ckpt}")

    config = {
        "headless": False,          # -> window="SDL2" in RedGymEnv
        "save_final_state": False,
        "print_rewards": True,
        "action_freq": 24,
        "init_state": args.init_state,
        "max_steps": 10_000_000,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path / "_watch",
        "gb_path": args.rom,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
    }

    # Training logged "Wrapping the env in a VecTransposeImage", so the saved
    # policy expects channels-first images. Replicate that wrapper or the
    # screens/map tensors arrive with the wrong layout.
    venv = VecTransposeImage(DummyVecEnv([lambda: RedGymEnv(config)]))
    for pb in venv.get_attr("pyboy"):
        pb.set_emulation_speed(args.speed)

    model = PPO.load(str(ckpt), env=venv, device=args.device)

    obs = venv.reset()
    step = 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, _, _ = venv.step(action)   # VecEnv auto-resets on done
            step += 1
            if args.reload_every and step % args.reload_every == 0:
                newest = latest_checkpoint(sess_path)
                if newest and newest != ckpt:
                    ckpt = newest
                    print(f"\nreloading {ckpt}")
                    model = PPO.load(str(ckpt), env=venv, device=args.device)
    except KeyboardInterrupt:
        pass
    finally:
        venv.close()


if __name__ == "__main__":
    main()
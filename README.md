# Train RL agents to play Pokémon Red

A fork of PWhiddy/PokemonRedExperiments, updated for PyBoy 2.7 / Gymnasium 0.29 / SB3 2.3 and tuned for NVIDIA Jetson (aarch64).

## Quick Start

**ROM**: place `PokemonRed.gb` (1 MB) in the repository root. `sha1sum` must be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`.

**System packages (Linux)**:

```bash
sudo apt install -y ffmpeg libsdl2-2.0-0
```

*ffmpeg* is needed for `save_video`; *libsdl2* for windowed viewing. The pip package `pysdl2-dll` only ships Windows/macOS binaries and does nothing on aarch64.

**Install**:

```bash
cd v2
pip install -r requirements.txt
pip install --no-deps stable_baselines3==2.3.2   # see warning below
```

**Verify** the environment before committing hours to a run:

```bash
python baseline_fast_v2.py --check-env
```

**Train**:

```bash
python baseline_fast_v2.py --num-cpu 12 --n-steps 1024 --device cuda
```

## ⚠️ Jetson / aarch64: torch must come from JetPack

`stable_baselines3` declares `torch>=1.13`, so a plain `pip install -r requirements.txt` will silently pull the CPU-only PyPI aarch64 wheel and shadow your JetPack build. Always verify:

```bash
python -c "import torch; print(torch.__file__, torch.version.cuda, torch.cuda.is_available())"
```

You want `torch.version.cuda` to be a version string and `cuda.is_available()` to be `True`. If `version.cuda` is `None`, you have the CPU wheel:

```bash
pip uninstall -y torch triton
python -c "import torch; print(torch.__file__, torch.version.cuda)"   # re-check
pip install --no-deps stable_baselines3==2.3.2
```

The JetPack wheel typically lives in `~/.local/lib/python3.x/site-packages/` (user site), not `/usr/lib/python3/dist-packages/` — searching only the latter will make it look uninstalled. A venv package always shadows both, so `--system-site-packages` will not save you.

> **Note**: Never pin `torch` or `triton` in `requirements.txt` on Jetson.

## Command-line reference

```bash
--num-cpu N        Parallel environments (default: min(8, core count))
--n-steps N        PPO rollout length per env (default 1024)
--ep-length N      Episode truncation length / max_steps (default 163840)
--batch-size N     PPO minibatch (default 512)
--n-epochs N       Passes per rollout (default 1; 3 is usually better)
--save-freq N      Checkpoint every N per-env steps (default 50000)
--checkpoint PATH  Resume from a .zip
--total-steps N    Stop after N timesteps (default 100M)
--watch            Render env 0 in an SDL2 window
--stream           Broadcast coords to the community map
--check-env        Run SB3's env_checker on one instance and exit
--device cuda|cpu  Torch device (default auto)
--rom / --init-state / --sess-id / --seed
```

### `--n-steps` and `--ep-length` are independent — and must be

These were coupled in the original (`n_steps = ep_length // num_cpu`), which forces one bad tradeoff or the other. They want opposite values:

| Wants | Why |
| :--- | :--- |
| **--ep-length** | Long | Enough in-game time to reach real objectives |
| **--n-steps** | Short | Buffer size and update frequency |

The rollout buffer is the constraint. SB3's `DictRolloutBuffer` stores observations as `float32` regardless of the space `dtype`, so it costs roughly `n_steps` × `num_cpu` × 83 KB:

| n_steps × num_cpu | Buffer | Update interval |
| :--- | :--- | :--- |
| 512 × 8 = 4,096 | ~340 MB | ~20 s |
| 1024 × 12 = 12,288 | ~1.0 GB | ~30 s |
| 2048 × 16 = 32,768 | ~2.7 GB | ~90 s |
| 20480 × 8 = 163,840 | ~13.6 GB | hours |

That last row is what the original config produced. It will OOM a 16 GB Orin NX and thrash a 64 GB AGX.

> **Tip**: Keep `n_steps` × `num_cpu` divisible by `--batch-size` or SB3 drops the remainder every update.

## How long should an episode be?

Not 24 hours. With `gamma=0.997`, the effective horizon is `1/(1-0.997) ≈ 333 steps`. Rewards much beyond ~1000 steps ahead are discounted into numerical irrelevance, so extra episode length buys no credit assignment.

What teaches the early game is repetition. Longer episodes mean fewer attempts at leaving the house. At `--ep-length 163840` each env restarts every ~25–30 min of wall clock — roughly 20 fresh attempts per env overnight. A 24-hour episode gives you one.

At `action_freq=24`, one agent step ≈ 0.4 s of game time:

| --ep-length | In-game time |
| :--- | :--- |
| 4,096 | ~27 min |
| 16,384 | ~1.8 h |
| 163,840 | ~18 h |

## You do not need to babysit and reload

Two different things get confused here:

| Restores | When |
| :--- | :--- |
| **Checkpoint (.zip)** | Policy weights — what was learned | Only on process restart |
| **Episode reset** | Game state back to init.state | Automatically, every max_steps |

Learning is continuous across episode resets. Resets don't undo progress — they are the practice mechanism. Start it once and leave it.

## Overnight run

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks    # MAXN, unlock clocks

tmux new -s poke
python baseline_fast_v2.py \
  --num-cpu 12 --ep-length 163840 --n-steps 1024 \
  --batch-size 1024 --n-epochs 3 --save-freq 50000 \
  --device cuda 2>&1 | tee runs/train_$(date +%m%d_%H%M).log
```

Detach with `Ctrl-B D`, reattach with `tmux attach -t poke`.

After ~10 minutes, read SB3's time/fps and multiply by your run duration in seconds — that's your real step budget. Measure it rather than trusting an estimate. Also check `du -sh runs/checkpoints/` early; if the zips are large, raise `--save-freq`.

`Ctrl-C` saves to `runs/poke_final.zip`.

## Resuming

```bash
python baseline_fast_v2.py --checkpoint runs/checkpoints/poke_1000000_steps.zip
```

`reset_num_timesteps=False` is applied automatically when resuming, so TensorBoard curves continue rather than restarting at zero.

## Performance expectations

The GPU will look idle. That is correct. PyBoy is pure CPU and single-threaded per env; the GPU only serves the policy update, which is microseconds for a ~3k-feature net. Throughput is env-bound.

`--num-cpu` should not exceed your physical core count (12 on AGX Orin). To actually load the GPU, widen the network — but only after the reward function is validated:

```python
policy_kwargs = dict(net_arch=[512, 512],
                     features_extractor_kwargs=dict(cnn_output_dim=512))
```

The original project's headline results came from ~40M+ steps aggregated across many contributors' machines. One Jetson overnight is a real contribution to that scale, not a replacement for it. At ~1M steps, expect bedroom fumbling — that's on track, not broken.

Historically, reward shaping mattered more than compute on this project. That's the higher-leverage thing to iterate on.

## Monitoring

### TensorBoard

```bash
cd v2 && tensorboard --logdir runs    # http://localhost:6006
```

Worth watching:

| Metric | Meaning |
| :--- | :--- |
| **rollout/ep_rew_mean** | Primary signal (requires VecMonitor, enabled) |
| **env_stats/coord_count** | Unique tiles seen — the real early-game progress measure |
| **env_stats/battle_wins_mean** | Validates the win-detection fix |
| **env_stats_max/*** | Best env, not the mean — where to look for breakthroughs |
| **trajectory/explore_sum** | Union of all envs' explored map |
| **time/fps** | Throughput |

`log_freq` / `image_freq` in `tensorboard_callback.py` count vec-env steps, not agent timesteps — at 12 envs, `log_freq=2048` is one sample per ~25k steps.

### Watching the agent play

Prefer the spectator script over `--watch`:

```bash
python watch_checkpoint.py --speed 3 --reload-every 20000
```

It loads the newest checkpoint and plays it in a window while training continues untouched. `--reload-every` picks up new checkpoints as they land. It runs inference only — no `model.learn()`, no gradients, no checkpoints written, no TensorBoard writes. Costs ~1 core, no GPU with `--device cpu`.

> **Why not --watch**: `SubprocVecEnv` steps all workers in lockstep, so throttling env 0 to a viewable speed rate-limits your entire rollout collection. It also requires restarting training to enable.

`--deterministic` makes behaviour easier to read but more prone to corner-stuck loops — itself informative about whether the stuck penalty is working.

Headless machine? Set `save_video: True` and collect mp4s from `runs/_watch/rollouts/`.

## Save states and PyBoy versions

PyBoy save states are version-locked. A state written by 2.4.0 will warn or fail under 2.7.0.

`RedGymEnv._load_initial_state()` handles this: on load failure it rebuilds the emulator, auto-skips the intro, and writes a fresh state. `baseline_fast_v2.py` calls `bootstrap_init_state()` in the parent before workers spawn, so N workers can't race on the same file.

The auto-skip mashes `START`/`A` blindly, which names the player and rival `AAAAAAA`. For a hand-picked start point, make your own state.

Seeing `Loading state from an older version of PyBoy`? It loaded, but re-save it natively to drop latent CPU/MBC drift:

```python
from pyboy import PyBoy
pb = PyBoy("../PokemonRed.gb", window="null")
with open("../init.state", "rb") as f: pb.load_state(f)
pb.tick(1, False)
with open("../init.state.new", "wb") as f: pb.save_state(f)
pb.stop(save=False)
```

Then `mv ../init.state.new ../init.state` and confirm the warning is gone.

Pin `pyboy==2.7.0` so a future `pip install -r` can't move you silently.

## Reward function

Defined in `red_gym_env_v2.py::get_game_state_reward()`.

Only monotonic terms belong there. `update_reward()` differences that dict each step, so any term that can decrease refunds its own reward and becomes farmable. Transient terms go in `get_instantaneous_reward()`.

| Term | Weight | Notes |
| :--- | :--- | :--- |
| **event** | ×4 | Max-tracked event flags |
| **level** | ×2 | Sub-linear above level-sum 22 |
| **heal** | ×10 | Linear in HP fraction healed |
| **badge** | ×10 | Popcount of 0xD356 |
| **explore** | ×0.1 | Unique coords |
| **pokedex** | ×2 | Popcount over owned flags |
| **battle** | ×0.5 | Cumulative battles entered |
| **win** | ×50 | Requires enemy HP to actually reach 0 |
| **stuck** | −0.05 | Instantaneous; ≥600 visits to one tile |

Bugs fixed in this fork, worth knowing if you compare against upstream:
- Pokedex read 0xD30A as a count when it's the start of a bitfield — producing a garbage 0–5100 term that dominated everything else.
- Battle / stuck were refundable, so cycling in and out of battle farmed reward.
- `battle_won_count` counted fleeing as a win, at 50 points each.
- Event-flag naming used MSB-first bit indices against LSB-first constants, so lookups never matched (the upstream # TODO this currently seems to be broken!).
- The cv2 telemetry overlay was drawn into `_get_obs()`, contaminating CNN input whenever `headless=False`.
- `get_explore_map()` wrapped negative slices near map edges.
- Headless never called `set_emulation_speed(0)`, so it ran at real time.
- `reset()` never loaded the save state — every episode continued from the previous one.

Changing weights invalidates a checkpoint's value function. The policy still loads; expect a temporary reward dip.

## Repository layout

| File | Role |
| :--- | :--- |
| `baseline_fast_v2.py` | Training entry point, CLI, vec-env setup |
| `red_gym_env_v2.py` | Gymnasium env: obs, rewards, memory reads |
| `tensorboard_callback.py` | Env telemetry → TensorBoard |
| `stream_agent_wrapper.py` | Optional coord broadcast (threaded, non-blocking) |
| `global_map.py` | local_to_global coordinate translation |
| `events.json` | Event-flag name lookup |
| `watch_checkpoint.py` | Inference-only spectator |

## Troubleshooting

| Symptom | Cause |
| :--- | :--- |
| Using `cpu` device | CPU-only torch wheel shadowing JetPack — see above |
| AttributeError: no attribute 'check_if_done' | Stale `tensorboard_callback.py`; clear `__pycache__` |
| module compiled with NumPy 1.x | Pin `numpy==1.26.4`, `pandas==2.1.4`, `scikit-image==0.22.0` |
| OOM at startup | `n_steps` × `num_cpu` too large — see buffer table |
| BrokenPipeError after a worker traceback | Cascade noise; the real error is the first traceback |
| No window with `--watch` | `apt install libsdl2-2.0-0`; check `$DISPLAY`; `ssh -X` |
| Video writer fails | `apt install ffmpeg` |
| Error in cpuinfo: prctl(PR_SVE_GET_VL) | Harmless `py-cpuinfo` quirk on ARM |
| Unable to import Axes3D | Harmless dual `matplotlib` install |
| `--num-cpu` has no effect on throughput | You're on `DummyVecEnv` (only used when `num_cpu == 1`) |

## Ideas worth pursuing

- **Intrinsic curiosity (RND)**: the highest-leverage change for early-game exploration — likely more impactful than any amount of extra compute.
- **Decouple gamma from episode length**: try `gamma=0.999` (horizon ~1000) now that episodes are long, and see whether longer-range credit assignment helps or destabilises the value function.
- **Battle strategy**: reward type-effective move selection rather than just win/loss.
- **Curriculum learning**: stage states at known checkpoints (post-starter, post-Brock) and train forward from each.
- **Event-triggered webhooks**: post a frame on badge gain or new max_map_progress instead of on a timer — a real progress feed rather than screenshot spam.

Original documentation and research references: see the upstream repository.

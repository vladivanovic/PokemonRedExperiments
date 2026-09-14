# Train RL agents to play Pokemon Red

## Quick Start (V2)

1. **Prepare ROM:** Place your `PokemonRed.gb` (1MB) in the **root** directory of this repository. 
   *(The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`)*.
2. **Install:**
   ```bash
   cd v2
   pip install -r requirements.txt
   ```
3. **Train:**
   ```bash
   python baseline_fast_v2.py
   ```

## Advanced Usage

### Continuing a training run
To resume from a checkpoint:
```bash
echo "runs/poke_1000000_steps" | python baseline_fast_v2.py
```
*(Ensure `v2/runs/poke_1000000_steps.zip` exists).*

### Visualizing Training
- **TensorBoard:** From `v2/` directory:
  ```bash
  tensorboard --logdir runs
  ```
  Open `http://localhost:6006`.
- **In-Game View:** Run the training script with the `--watch` flag to see the first environment:
  ```bash
  python baseline_fast_v2.py --watch
  ```

### Improving the Agent
- **Reward Function:** Edit `v2/red_gym_env_v2.py` to add custom reward signals.
- **Hyperparameters:** Modify `baseline_fast_v2.py` to tune PPO parameters.

## Future Updates to Pursue
- **Intrinsic Curiosity (RND):** Implement Random Network Distillation to improve exploration and prevent the agent from getting stuck.
- **Battle Strategy:** Enhance battle detection to specifically reward "Super Effective" move usage and advanced combat strategies.
- **Curriculum Learning:** Introduce staged training scenarios (e.g., pre-leveled starters or specific game progress points) to master complex game sequences.
- **Telemetry Overlay:** Further refine the in-game UI to provide more detailed real-time performance insights during `--watch` mode.

---
*For original documentation and research references, see the original README.*

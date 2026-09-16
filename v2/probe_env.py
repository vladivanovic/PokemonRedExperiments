# probe_env.py
from pathlib import Path
from red_gym_env_v2 import RedGymEnv, TEXTBOX_ID, PARTY_COUNT

env = RedGymEnv({
    "gb_path": "../PokemonRed.gb", "init_state": "../init.state",
    "session_path": Path("runs/_probe"), "headless": True,
    "max_steps": 100000, "reward_scale": 0.5, "explore_weight": 1.0,
})
env.reset()
tb_nonzero = 0
for i in range(600):
    env.step(env.action_space.sample())
    if env.read_m(TEXTBOX_ID) != 0:
        tb_nonzero += 1
    if i % 200 == 0:
        print(f"step {i:4d} textbox={env.read_m(TEXTBOX_ID):3d} "
              f"coords={len(env.seen_coords):4d} "
              f"party={env.read_m(PARTY_COUNT)} "
              f"term={env.check_terminated()} "
              f"twipe={env.terminate_on_wipe}")
print(f"\ntextbox nonzero on {tb_nonzero}/600 steps")
print(f"final seen_coords={len(env.seen_coords)}")
env.close()
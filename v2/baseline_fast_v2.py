import sys
import argparse
from os.path import exists
from pathlib import Path
from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0, watch=False):
    def _init():
        config = env_conf.copy()
        if watch and rank == 0:
            config['headless'] = False
        
        env = StreamWrapper(
            RedGymEnv(config), 
            stream_metadata = { 
                "user": "v2-default", 
                "env_id": rank, 
                "color": "#447799", 
                "extra": "",
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="Visualize the first environment")
    parser.add_argument("--num_cpu", type=int, default=64, help="Number of CPU cores to use")
    args = parser.parse_args()

    use_wandb_logging = False
    ep_length = 2048 * 80
    sess_id = "runs"
    sess_path = Path(sess_id)

    env_config = {
                'headless': True, 'save_final_state': False, 'early_stop': False,
                'action_freq': 24, 'init_state': '../init.state', 'max_steps': ep_length, 
                'print_rewards': True, 'save_video': False, 'fast_video': True, 'session_path': sess_path,
                'gb_path': '../PokemonRed.gb', 'debug': False, 'reward_scale': 0.5, 'explore_weight': 0.25
            }
    
    print(f"Starting training with {args.num_cpu} CPUs, Watch mode: {args.watch}")
    
    env = SubprocVecEnv([make_env(i, env_config, watch=args.watch) for i in range(args.num_cpu)])
    
    checkpoint_callback = CheckpointCallback(save_freq=ep_length//2, save_path=sess_path,
                                     name_prefix="poke")
    
    callbacks = [checkpoint_callback, TensorboardCallback(sess_path)]

    if sys.stdin.isatty():
        file_name = ""
    else:
        file_name = sys.stdin.read().strip()

    train_steps_batch = ep_length // args.num_cpu
    
    if exists(file_name + ".zip"):
        print("\nloading checkpoint")
        model = PPO.load(file_name, env=env)
        model.n_steps = train_steps_batch
        model.n_envs = args.num_cpu
        model.rollout_buffer.buffer_size = train_steps_batch
        model.rollout_buffer.n_envs = args.num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO("MultiInputPolicy", env, verbose=1, n_steps=train_steps_batch, batch_size=512, n_epochs=1, gamma=0.997, ent_coef=0.01, tensorboard_log=sess_path)
    
    print(model.policy)

    model.learn(total_timesteps=(ep_length)*args.num_cpu*10000, callback=CallbackList(callbacks), tb_log_name="poke_ppo")

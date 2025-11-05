

import os
import argparse
import distutils.util
from multiprocessing import cpu_count

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback



try:
    from botv2 import BattleBotsEnv, TESTING_KWARGS, str_to_bool
except ImportError:
    print("Error: Could not import from botv2.py.")
    print("Please make sure finetune.py is in the same directory as main.py")
    exit(1)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Finetune a PPO agent for BattleBots.")
    
    
    parser.add_argument('--model_to_finetune', type=str, 
                        default="./ppo_battlebots_tensorboard/ppo_battlebots_agent_v2/best_model/best_model.zip",
                        help='Path to the "best_model.zip" from the initial training run.')
    
    parser.add_argument('--new_model_name', type=str, 
                        default="ppo_battlebots_finetuned_v1.zip",
                        help='Filename for saving the new *finetuned* model.')
    
    parser.add_argument('--log_dir', type=str, 
                        default="./ppo_battlebots_tensorboard/",
                        help='Tensorboard log directory. Finetuning logs will go into a subfolder here.')

    
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='A *small* learning rate for finetuning (e.g., 1e-5 or 5e-6).')
    
    parser.add_argument('--total_timesteps', type=int, default=300_000,
                        help='Number of timesteps to finetune for.')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible finetuning.')

    args = parser.parse_args()

    
    if not os.path.exists(args.model_to_finetune):
        print(f"Error: Model file not found at '{args.model_to_finetune}'")
        print("Please run main.py to train a model first, or check the path.")
        exit(1)

    
    num_cpu = max(1, cpu_count() - 1) * 2
    
    vec_env_kwargs = {"env_kwargs": TESTING_KWARGS}
    
    if args.seed is not None:
        print(f"--- Setting random seed to: {args.seed} ---")
        set_random_seed(args.seed)
        vec_env_kwargs['seed'] = args.seed
    else:
        print("--- No seed set, running non-deterministically ---")

    
    run_log_dir = os.path.join(args.log_dir, os.path.splitext(args.new_model_name)[0])
    print(f"Finetuning logs will be saved to: {run_log_dir}")

    
    print(f"--- Creating Finetuning Environment (using TESTING_KWARGS) with {num_cpu} cores ---")
    finetune_env = make_vec_env(BattleBotsEnv, n_envs=num_cpu,
                                vec_env_cls=SubprocVecEnv, 
                                **vec_env_kwargs)

    
    print("Creating evaluation environment...")
    eval_kwargs = {"env_kwargs": TESTING_KWARGS}
    if args.seed is not None:
        eval_kwargs['seed'] = args.seed + 1 
    
    eval_env = make_vec_env(BattleBotsEnv, n_envs=1, **eval_kwargs)

    
    print(f"--- Loading model from: {args.model_to_finetune} ---")
    model = PPO.load(args.model_to_finetune)

    
    print(f"--- Re-configuring model for finetuning ---")
    
    
    model.set_env(finetune_env)
    
    
    model.learning_rate = args.lr
    print(f"  Set new learning rate to: {model.learning_rate}")
    
    
    print(f"  Set new TensorBoard log dir to: {run_log_dir}")

    
    
    best_model_save_dir = os.path.join(run_log_dir, "best_model")
    print(f"Best *finetuned* model will be saved to: {best_model_save_dir}")

    eval_callback = EvalCallback(eval_env, 
                             best_model_save_path=best_model_save_dir,
                             log_path=best_model_save_dir, 
                             eval_freq=2500,   
                             n_eval_episodes=5, 
                             deterministic=True,
                             render=False)

    
    print(f"\n--- Starting Finetuning for {args.total_timesteps} timesteps ---")
    model.learn(total_timesteps=args.total_timesteps, 
                progress_bar=True, 
                callback=eval_callback,
                reset_num_timesteps=False) 
    
    
    model.save(args.new_model_name)
    
    print("\n--- Finetuning Finished ---")
    print(f"Final finetuned model saved to: {args.new_model_name}")
    print(f"Best finetuned model saved in: {best_model_save_dir}")
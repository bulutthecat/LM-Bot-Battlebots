# finetune.py

import os
import argparse
import distutils.util
from multiprocessing import cpu_count

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback

# --- Import from our main script ---
# This requires the script to be in the same directory as main.py
try:
    from botv2 import BattleBotsEnv, TESTING_KWARGS, str_to_bool
except ImportError:
    print("Error: Could not import from botv2.py.")
    print("Please make sure finetune.py is in the same directory as main.py")
    exit(1)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Finetune a PPO agent for BattleBots.")
    
    # --- File/Log Management ---
    parser.add_argument('--model_to_finetune', type=str, 
                        default="./ppo_battlebots_tensorboard/ppo_battlebots_agent_v2/best_model/best_model.zip",
                        help='Path to the "best_model.zip" from the initial training run.')
    
    parser.add_argument('--new_model_name', type=str, 
                        default="ppo_battlebots_finetuned_v1.zip",
                        help='Filename for saving the new *finetuned* model.')
    
    parser.add_argument('--log_dir', type=str, 
                        default="./ppo_battlebots_tensorboard/",
                        help='Tensorboard log directory. Finetuning logs will go into a subfolder here.')

    # --- Finetuning Hyperparameters ---
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='A *small* learning rate for finetuning (e.g., 1e-5 or 5e-6).')
    
    parser.add_argument('--total_timesteps', type=int, default=300_000,
                        help='Number of timesteps to finetune for.')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible finetuning.')

    args = parser.parse_args()

    # --- Check if model exists ---
    if not os.path.exists(args.model_to_finetune):
        print(f"Error: Model file not found at '{args.model_to_finetune}'")
        print("Please run main.py to train a model first, or check the path.")
        exit(1)

    # --- Set up CPU count and Seeding ---
    num_cpu = max(1, cpu_count() - 1) * 2
    
    vec_env_kwargs = {"env_kwargs": TESTING_KWARGS}
    
    if args.seed is not None:
        print(f"--- Setting random seed to: {args.seed} ---")
        set_random_seed(args.seed)
        vec_env_kwargs['seed'] = args.seed
    else:
        print("--- No seed set, running non-deterministically ---")

    # --- Create a unique log directory for this finetuning run ---
    run_log_dir = os.path.join(args.log_dir, os.path.splitext(args.new_model_name)[0])
    print(f"Finetuning logs will be saved to: {run_log_dir}")

    # --- 1. Create the "Harder" Finetuning Environment ---
    print(f"--- Creating Finetuning Environment (using TESTING_KWARGS) with {num_cpu} cores ---")
    finetune_env = make_vec_env(BattleBotsEnv, n_envs=num_cpu,
                                vec_env_cls=SubprocVecEnv, 
                                **vec_env_kwargs)

    # --- 2. Create a separate Evaluation Environment ---
    print("Creating evaluation environment...")
    eval_kwargs = {"env_kwargs": TESTING_KWARGS}
    if args.seed is not None:
        eval_kwargs['seed'] = args.seed + 1 # Use a different seed for eval
    
    eval_env = make_vec_env(BattleBotsEnv, n_envs=1, **eval_kwargs)

    # --- 3. Load the Pre-trained Model ---
    print(f"--- Loading model from: {args.model_to_finetune} ---")
    model = PPO.load(args.model_to_finetune)

    # --- 4. Re-configure the Model for Finetuning ---
    print(f"--- Re-configuring model for finetuning ---")
    
    # Set the new (harder) environment
    model.set_env(finetune_env)
    
    # Set the new (smaller) learning rate
    model.learning_rate = args.lr
    print(f"  Set new learning rate to: {model.learning_rate}")
    
    # Set the new TensorBoard log directory
    print(f"  Set new TensorBoard log dir to: {run_log_dir}")

    # --- 5. Create a New Evaluation Callback ---
    # This will save the *best finetuned* model to a new location
    best_model_save_dir = os.path.join(run_log_dir, "best_model")
    print(f"Best *finetuned* model will be saved to: {best_model_save_dir}")

    eval_callback = EvalCallback(eval_env, 
                             best_model_save_path=best_model_save_dir,
                             log_path=best_model_save_dir, 
                             eval_freq=2500,   # Check every 10,000 steps
                             n_eval_episodes=5, # Run 10 episodes for a more stable eval
                             deterministic=True,
                             render=False)

    # --- 6. Run Finetuning (model.learn) ---
    print(f"\n--- Starting Finetuning for {args.total_timesteps} timesteps ---")
    model.learn(total_timesteps=args.total_timesteps, 
                progress_bar=True, 
                callback=eval_callback,
                reset_num_timesteps=False) # <<< IMPORTANT: Continues step count
    
    # --- 7. Save the Final Finetuned Model ---
    model.save(args.new_model_name)
    
    print("\n--- Finetuning Finished ---")
    print(f"Final finetuned model saved to: {args.new_model_name}")
    print(f"Best finetuned model saved in: {best_model_save_dir}")
import os
import numpy as np
from flask import Flask, request, jsonify
from stable_baselines3 import PPO



MODEL_FILENAME = "best_model.zip"



ARENA_WIDTH = 1000.0
ARENA_HEIGHT = 700.0
ARENA_TOP = 10.0
ARENA_LEFT = 0.0
BOT_RADIUS = 13.0
BOT_DIAMETER = BOT_RADIUS * 2.0
MAX_BULLETS_PER_BOT = 4



TESTING_KWARGS = {
    'max_bots': 16,
}



MAX_BOTS = TESTING_KWARGS['max_bots']
MAX_TOTAL_BULLETS = (MAX_BOTS + 1) * MAX_BULLETS_PER_BOT


MIN_X = ARENA_LEFT
MAX_X = ARENA_WIDTH - BOT_DIAMETER
MIN_Y = ARENA_TOP
MAX_Y = ARENA_HEIGHT - BOT_DIAMETER


app = Flask(__name__)
model = None
g_prev_obs = None 

def _build_obs_from_json(data: dict) -> np.ndarray:
    """
    Converts the JSON data from the Java client into the normalized,
    flattened observation vector that the model expects.
    
    This is a re-implementation of the _get_obs() logic from your Env.
    """
    
    
    agent_pos = data.get('agent_pos', [0.0, 0.0])
    agent_pos_norm = np.array([
        (agent_pos[0] - MIN_X) / (MAX_X - MIN_X) * 2 - 1,
        (agent_pos[1] - MIN_Y) / (MAX_Y - MIN_Y) * 2 - 1
    ], dtype=np.float32)
    
    
    agent_ammo = data.get('agent_ammo', 0)
    agent_ammo_norm = np.array([(agent_ammo / MAX_BULLETS_PER_BOT) * 2 - 1], dtype=np.float32)

    
    other_bots = data.get('other_bots', [])
    num_bots_alive = len(other_bots)
    bot_info_padded = np.full((MAX_BOTS, 3), -1.0, dtype=np.float32)
    
    if num_bots_alive > 0:
        
        live_bots_data = np.array(other_bots[:MAX_BOTS], dtype=np.float32)
        
        
        bot_pos_norm_x = (live_bots_data[:, 0] - MIN_X) / (MAX_X - MIN_X) * 2 - 1
        bot_pos_norm_y = (live_bots_data[:, 1] - MIN_Y) / (MAX_Y - MIN_Y) * 2 - 1
        
        
        bot_ammo_norm = (live_bots_data[:, 2] / MAX_BULLETS_PER_BOT) * 2 - 1
        
        bot_info_padded[:num_bots_alive, 0] = bot_pos_norm_x
        bot_info_padded[:num_bots_alive, 1] = bot_pos_norm_y
        bot_info_padded[:num_bots_alive, 2] = bot_ammo_norm
        
    
    bullets = data.get('bullets', [])
    num_bullets_active = len(bullets)
    bullet_info_padded = np.full((MAX_TOTAL_BULLETS, 2), -1.0, dtype=np.float32)
    
    if num_bullets_active > 0:
        
        active_bullets_data = np.array(bullets[:MAX_TOTAL_BULLETS], dtype=np.float32)
        
        
        bullet_pos_norm_x = (active_bullets_data[:, 0] - MIN_X) / (ARENA_WIDTH - MIN_X) * 2 - 1
        bullet_pos_norm_y = (active_bullets_data[:, 1] - MIN_Y) / (ARENA_HEIGHT - MIN_Y) * 2 - 1
        
        bullet_info_padded[:num_bullets_active, 0] = bullet_pos_norm_x
        bullet_info_padded[:num_bullets_active, 1] = bullet_pos_norm_y

    
    current_obs = np.concatenate([
        agent_pos_norm, 
        agent_ammo_norm, 
        bot_info_padded.flatten(), 
        bullet_info_padded.flatten()
    ]).astype(np.float32)
    
    return current_obs

@app.route("/reset", methods=["POST"])
def reset_state():
    """
    Resets the server's observation state.
    Call this at the beginning of each new episode.
    It expects the *first* frame of data and returns the *first* action.
    """
    global g_prev_obs
    
    try:
        data = request.json
        
        current_obs = _build_obs_from_json(data)
        
        
        g_prev_obs = current_obs
        
        
        stacked_obs = np.concatenate([g_prev_obs, current_obs])
        
        
        action, _ = model.predict(stacked_obs, deterministic=True)
        
        return jsonify({"action": int(action)})
        
    except Exception as e:
        print(f"Error in /reset: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict_action():
    """
    Predicts an action for the current frame.
    Call this for every subsequent frame after /reset.
    It automatically stacks the new frame with the previous one.
    """
    global g_prev_obs
    
    if g_prev_obs is None:
        return jsonify({"error": "State is not initialized. Call /reset first."}), 400
    
    try:
        data = request.json
        
        current_obs = _build_obs_from_json(data)
        
        
        stacked_obs = np.concatenate([g_prev_obs, current_obs])
        
        
        g_prev_obs = current_obs
        
        
        action, _ = model.predict(stacked_obs, deterministic=True)
        
        return jsonify({"action": int(action)})

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(MODEL_FILENAME):
        print(f"Error: Model file '{MODEL_FILENAME}' not found.")
        print("Please train the model first by running main.py with TRAIN_MODEL=True")
    else:
        print(f"--- Loading model {MODEL_FILENAME}... ---")
        model = PPO.load(MODEL_FILENAME)
        print("--- Model loaded. Starting server... ---")
        
        print("\nServer is running. Your Java program can now send requests to:")
        print("  POST http://127.0.0.1:5000/reset   (at the start of an episode)")
        print("  POST http://127.0.0.1:5000/predict (for every frame after)")
        
        
        
        app.run(host='127.0.0.1', port=5000, debug=False)
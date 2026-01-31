import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
from gymnasium.spaces import Box
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import ale_py


# --- 1. Custom Aggregation Layer for Loading ---
def combine_layer(args):
    v, a = args
    return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))


# --- 2. Setup Environment ---
# Note: terminal_on_life_loss is True to match training observation flow,
# but we will handle the "reset" logic manually to keep the 5-life game going.
env = gym.make("ALE/Breakout-v5", render_mode="human", frameskip=1)
env = AtariPreprocessing(env, noop_max=30, frame_skip=4, terminal_on_life_loss=False, grayscale_obs=True)
env = FrameStackObservation(env, stack_size=4)

new_obs_space = Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
env = TransformObservation(env, lambda obs: np.moveaxis(obs, 0, -1), observation_space=new_obs_space)

# --- 3. Load the Model ---
checkpoint_path = "../Model/breakout_dqn_final.keras"
model = keras.models.load_model(
    checkpoint_path,
    custom_objects={'combine_layer': combine_layer},
    safe_mode=False
)


def play_episodes(num_episodes=5):
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        # Track lives to detect when the ball is lost
        # ALE starts with 5 lives in Breakout
        current_lives = info.get('lives', 5)

        print(f"--- Starting Episode {ep + 1} ---")

        # Initial FIRE to start the ball
        obs, reward, terminated, truncated, info = env.step(1)

        while not (terminated or truncated):
            # 1. Prepare state (uint8 is fine as the model has a Rescaling layer)
            state_tensor = tf.convert_to_tensor(obs)
            state_tensor = tf.expand_dims(state_tensor, 0)

            # 2. Predict Action
            q_values = model(state_tensor, training=False)
            print(q_values.numpy())
            action = tf.argmax(q_values[0]).numpy()

            # 3. Step Environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # 4. Handle Life Loss (The "Fire" Trap)
            # If terminal_on_life_loss is True, the env returns terminated=True on life loss.
            # We check the 'lives' in info to see if we should actually stop or just re-fire.
            new_lives = info.get('lives', current_lives)

            # IF LIVES DROPPED: Re-fire automatically
            if new_lives < current_lives and not (terminated or truncated):
                print(f"Life lost! {new_lives} remaining. Re-firing...")
                # In Breakout, losing a life pauses the game.
                # We need to send Action 1 (FIRE) to continue.
                obs, _, _, _, info = env.step(1)
                current_lives = new_lives

            # Slow down for human eyes
            time.sleep(0.01)

        print(f"Episode {ep + 1} Finished. Total Reward: {total_reward}")


# Run it
try:
    play_episodes()
finally:
    env.close()
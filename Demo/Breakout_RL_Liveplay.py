import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformObservation
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import ale_py


# --- 1. XAI Function ---

def compute_saliency_map(model, state_input):
    state_tensor = tf.convert_to_tensor(state_input, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(state_tensor)
        q_values = model(state_tensor)
        max_q_value = tf.reduce_max(q_values, axis=1)

    grads = tape.gradient(max_q_value, state_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    return saliency.numpy()


# --- 2. Environment Setup ---

def combine_layer(args):
    v, a = args
    return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))


#Using rgb_array so we can manipulate the frames
env = gym.make("ALE/Breakout-v5", render_mode="rgb_array", frameskip=1)
env = AtariPreprocessing(env, terminal_on_life_loss=False)
env = FrameStackObservation(env, stack_size=4)
env = TransformObservation(env, lambda obs: np.moveaxis(obs, 0, -1),
                           observation_space=gym.spaces.Box(0, 255, (84, 84, 4), np.uint8))

# Load Model
model = keras.models.load_model("../Model/breakout_dqn_final.keras",
                                custom_objects={'combine_layer': combine_layer}, safe_mode=False)


# --- 3. Play Loop ---

def play_with_xai():
    obs, info = env.reset()
    current_lives = info.get('lives', 5)
    cv2.namedWindow("DDQN XAI Dashboard", cv2.WINDOW_NORMAL)

    while True:
        state_tensor = np.expand_dims(obs, 0)

        # 1. Get Brain Data
        q_values = model.predict(state_tensor, verbose=0)
        action = np.argmax(q_values[0])
        saliency = compute_saliency_map(model, state_tensor)

        # 2. Get Visual Data
        raw_frame = env.render()  # Returns RGB array
        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

        # 3. Create Heatmap Overlay
        saliency_res = cv2.resize(saliency, (raw_frame.shape[1], raw_frame.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * saliency_res), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(raw_frame, 0.6, heatmap, 0.4, 0)


        # 4. Stitch together [Game | Heatmap ]
        dashboard = np.hstack((raw_frame, overlay))

        # Display
        cv2.imshow("DDQN XAI Dashboard", dashboard)

        # Step Env
        obs, reward, terminated, truncated, info = env.step(action)

        # Life loss logic
        new_lives = info.get('lives', current_lives)
        if new_lives < current_lives:
            obs, _, _, _, info = env.step(1)  # Fire
            current_lives = new_lives

        if (terminated or truncated) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    env.close()
    cv2.destroyAllWindows()


play_with_xai()
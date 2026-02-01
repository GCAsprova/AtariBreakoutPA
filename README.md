#Atari Breakout: Double DQN (DDQN) Agent

This project implements a Double Deep Q-Network to master the Atari Breakout environment using gymnasium and TensorFlow. The agent learns to map raw pixel data directly to paddle movements to maximize its game score.

#1. Project Files

Training_Notebook.ipynb: The training pipeline. Handles environment wrapping, the Replay Buffer,the training loop and model architecture.

LivePlay_XAI.py: The demonstration script. Loads the trained model and runs a "Dashboard" view with real-time Explainable AI (Saliency Map).

#2. How to Run

Training: 
  Install the dependencies via requirements.txt. The project was initially run in Google Colab. The Colab Environment can be recreated using Colab_env_requirements.txt.
  Run all the cells.
  The model and all checkpoints will be safed to your Google Drive. When run locally saving needs to be adjusted as the Google package is not available outside of Colab.

Demo:
  Install the dependencies via requirements.txt.
  Run the .py File.
  Window of the Agent playing and the saliency heat map will open. Can be prematurely stopped by pressing "q".

#3. Notes
  Model: 
    Was trained for 5 Million Frames.
    Manages to score an average of 44 Points.
    Hasn't discovered tunneling yet.

  Training parameters:
    Trained for the first 1.5 Million frames with :
        batch_size = 32
        learning_rate = 1e-4
        target_update_freq = 10_000   
        terminal_on_life_loss = True
        
    Training until 3.5 Million frames with
        batch_size = 64
        learning_rate = 1e-4
        target_update_freq = 15_000   
        terminal_on_life_loss = True
        
    Training until 5 Million frames with :
        batch_size = 64
        learning_rate = 0.5e-4
        target_update_freq = 15_000   
        terminal_on_life_loss = False

  Additional:
    ale_py package is never actively used, the environment somehow doesn't work without importing.
    

        
  
  

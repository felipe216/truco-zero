from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from truco.env import TrucoEnv
import os
import time # Import time for measuring TPS
from train.reward_callback import RewardLoggingCallback

def train_self_play(total_timesteps=200_000, log_interval=10000, save_interval=10_000, opponent_update_interval=10000):
    # Create environments for both agents
    # Each agent interacts with its own environment instance
    env_agent1 = TrucoEnv(mode="two_agents")
    env_agent1 = DummyVecEnv([lambda: Monitor(env_agent1)]) # Wrap the environment

    env_agent2 = TrucoEnv(mode="two_agents")
    env_agent2 = DummyVecEnv([lambda: Monitor(env_agent2)]) # Wrap the environment

    callback = RewardLoggingCallback()

    # Initialize PPO models for both agents
    model_agent1 = PPO("MlpPolicy", env_agent1, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    # Load previous model if resuming training
    try:
        model_agent1.load("models/agent1/ppo_truco_agent1_final", env=env_agent1)
        print("Loaded previous model for Agent 1.")
    except Exception as e:
        print(f"Could not load model for Agent 1: {e}. Starting from scratch.")

    model_agent2 = PPO("MlpPolicy", env_agent2, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    
    print("Iniciando treinamento self-play...")

    os.makedirs("./models/agent1", exist_ok=True)
    os.makedirs("./models/agent2", exist_ok=True)

    # Initial update of agent2 to be a copy of agent1 (or load a specific opponent)
    # This ensures agent1 starts training against a known policy.
    model_agent1.save("./temp_agent1_for_opponent.zip")
    model_agent2 = PPO.load("./temp_agent1_for_opponent.zip", env=env_agent2)
    print("Agente 2 inicializado como cópia do Agente 1.")


    current_timesteps = 0
    start_time = time.time()

    while current_timesteps < total_timesteps:
        # Update opponent periodically
        if current_timesteps > 0 and current_timesteps % opponent_update_interval == 0:
            model_agent1.save("./temp_agent1_for_opponent.zip")
            model_agent2 = PPO.load("./temp_agent1_for_opponent.zip", env=env_agent2)
            print(f"Agente 2 atualizado para a política do Agente 1 em {current_timesteps} timesteps.")

        # Train Agent 1 for n_steps
        # PPO's learn method will handle collecting n_steps and updating the model
        # The environment env_agent1 will interact with env_agent2 through the game logic
        # and the current_player mechanism within TrucoEnv.
        model_agent1.learn(total_timesteps=model_agent1.n_steps, reset_num_timesteps=False, callback=callback)
        current_timesteps += model_agent1.n_steps

        # Log progress
        if current_timesteps % log_interval == 0:
            elapsed_time = time.time() - start_time
            tps = current_timesteps / elapsed_time if elapsed_time > 0 else 0
            print(f"Timesteps: {current_timesteps}/{total_timesteps}, TPS: {tps:.2f}")

        # Save models
        if current_timesteps % save_interval == 0:
            model_agent1.save(f"./models/agent1/ppo_truco_agent1_{current_timesteps}")
            # Save opponent model as well, or just rely on the periodic update
            # model_agent2.save(f"./models/agent2/ppo_truco_agent2_{current_timesteps}")
            print(f"Modelos salvos em {current_timesteps} timesteps.")

    model_agent1.save("./models/agent1/ppo_truco_agent1_final")
    model_agent2.save("./models/agent2/ppo_truco_agent2_final") # Save final opponent
    print("Treinamento self-play finalizado.")

if __name__ == "__main__":
    # Example usage: train for 1 million timesteps, update opponent every 50k timesteps
    train_self_play(total_timesteps=200_000, log_interval=10000, save_interval=100000, opponent_update_interval=50000)
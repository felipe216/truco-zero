from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from truco.env import TrucoEnv
import os

def train_self_play(total_timesteps=1_000_000, log_interval=1000, save_interval=100_000):
    env_agent1 = TrucoEnv(mode="two_agents")
    env_agent1 = DummyVecEnv([lambda: env_agent1])

    env_agent2 = TrucoEnv(mode="two_agents")
    env_agent2 = DummyVecEnv([lambda: env_agent2])

    model_agent1 = PPO("MlpPolicy", env_agent1, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    model_agent1.load("models/ppo_truco", env=env_agent1)

    model_agent2 = PPO("MlpPolicy", env_agent2, verbose=0, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01)
    model_agent2.load("models/ppo_truco", env=env_agent2)    

    print("Iniciando treinamento self-play...")

    # Criar diretórios para salvar os modelos
    os.makedirs("./models/agent1", exist_ok=True)
    os.makedirs("./models/agent2", exist_ok=True)

    # Loop de treinamento
    obs_agent1 = env_agent1.reset()
    obs_agent2 = env_agent2.reset()

    current_timesteps = 0
    while current_timesteps < total_timesteps:
        # Obter o jogador atual do ambiente (para saber quem deve agir)
        # Acessamos o ambiente não-wrapped para obter o current_player
        current_player = env_agent1.envs[0].current_player

        if current_player == 1:
            # Agente 1 age
            action_agent1, _states = model_agent1.predict(obs_agent1, deterministic=False)
            obs_agent1, reward_agent1, done_agent1, info_agent1 = env_agent1.step(action_agent1)
            # O Agente 1 aprende com sua experiência
            model_agent1.learn(total_timesteps=1, reset_num_timesteps=False)
            
            # Se o jogo não terminou, é a vez do Agente 2
            if not done_agent1:
                # Agente 2 age (usando seu próprio modelo)
                action_agent2, _states = model_agent2.predict(obs_agent2, deterministic=False)
                obs_agent2, reward_agent2, done_agent2, info_agent2 = env_agent2.step(action_agent2)
                # O Agente 2 aprende com sua experiência
                model_agent2.learn(total_timesteps=1, reset_num_timesteps=False)
                
                # Se o jogo terminou após a ação do Agente 2, resetar ambos os ambientes
                if done_agent2:
                    obs_agent1 = env_agent1.reset()
                    obs_agent2 = env_agent2.reset()
            else:
                # Se o jogo terminou após a ação do Agente 1, resetar ambos os ambientes
                obs_agent1 = env_agent1.reset()
                obs_agent2 = env_agent2.reset()

        else: # current_player == 2
            # Agente 2 age
            action_agent2, _states = model_agent2.predict(obs_agent2, deterministic=False)
            obs_agent2, reward_agent2, done_agent2, info_agent2 = env_agent2.step(action_agent2)
            model_agent2.learn(total_timesteps=1, reset_num_timesteps=False)

            if not done_agent2:
                # Agente 1 age
                action_agent1, _states = model_agent1.predict(obs_agent1, deterministic=False)
                obs_agent1, reward_agent1, done_agent1, info_agent1 = env_agent1.step(action_agent1)
                model_agent1.learn(total_timesteps=1, reset_num_timesteps=False)

                if done_agent1:
                    obs_agent1 = env_agent1.reset()
                    obs_agent2 = env_agent2.reset()
            else:
                obs_agent1 = env_agent1.reset()
                obs_agent2 = env_agent2.reset()

        current_timesteps += 1

        if current_timesteps % log_interval == 0:
            print(f"Timesteps: {current_timesteps}/{total_timesteps}")

        if current_timesteps % save_interval == 0:
            model_agent1.save(f"./models/agent1/ppo_truco_agent1_{current_timesteps}")
            model_agent2.save(f"./models/agent2/ppo_truco_agent2_{current_timesteps}")
            print(f"Modelos salvos em {current_timesteps} timesteps.")

    model_agent1.save("./models/agent1/ppo_truco_agent1_final")
    model_agent2.save("./models/agent2/ppo_truco_agent2_final")
    print("Treinamento self-play finalizado.")

if __name__ == "__main__":
    train_self_play(total_timesteps=10000, log_interval=1000, save_interval=5000)
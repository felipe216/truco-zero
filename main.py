import argparse
import time
from train.train_agent import train_agent
from eval.player_vs_agent import play_game
from eval.player_vs_player import play_game_vs
from train.train_self_play import train_self_play
#from eval.play_vs_random import simular_jogo
#from eval.play_vs_model import jogar_contra_modelo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Projeto Truco com IA")
    parser.add_argument("--modo", choices=["treinar", "avaliar", "jogar", "contra"], required=True)
    args = parser.parse_args()

    if args.modo == "treinar":
        start_time = time.time()
        train_self_play()
        end_time = time.time()
        time_taken = end_time - start_time
        tps = 50000 / time_taken

        print(f"Tempo gasto para 50.000 timesteps: {time_taken:.2f} segundos")
        print(f"Taxa de Timesteps por Segundo (TPS): {tps:.2f}")

        estimated_time_for_1M_seconds = 1_000_000 / tps
        estimated_time_for_1M_hours = estimated_time_for_1M_seconds / 3600

        print(f"Tempo estimado para 1.000.000 timesteps: {estimated_time_for_1M_seconds:.2f} segundos ({estimated_time_for_1M_hours:.2f} horas)")
    #elif args.modo == "avaliar":
    #    simular_jogo()
    elif args.modo == "jogar":
        play_game()
    elif args.modo == "contra":
        play_game_vs()
    #else:
    #    raise ValueError(f"Modo inválido: {args.modo}")
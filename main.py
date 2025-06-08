import argparse
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
        train_self_play()
    #elif args.modo == "avaliar":
    #    simular_jogo()
    elif args.modo == "jogar":
        play_game()
    elif args.modo == "contra":
        play_game_vs()
    #else:
    #    raise ValueError(f"Modo inválido: {args.modo}")
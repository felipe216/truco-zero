# Truco Zero

## Overview
This project implements a reinforcement learning framework for the card game Truco. Inspired by the AlphaZero methodology, "Truco Zero" aims to develop intelligent agents capable of playing and mastering the game of Truco through self-play and advanced machine learning techniques.

## Features
- **Truco Game Environment**: A robust implementation of the Truco game rules, including bidding, card play, and score tracking.
- **Reinforcement Learning Agents**: Framework for developing and integrating various AI agents.
- **Self-Play Training**: Modules for training agents through self-play, generating experience data.
- **Evaluation Tools**: Utilities to evaluate agent performance against other agents or human players.
- **Model Checkpointing**: Mechanisms to save and load trained agent models.

## Project Structure
- `agents/`: Contains implementations of different AI agents (e.g., `random_agent.py`).
- `checkpoints/`: Stores saved models and training progress checkpoints.
- `eval/`: Scripts for evaluating agent performance (e.g., `player_vs_agent.py`, `player_vs_player.py`, `player_vs_random.py`).
- `models/`: Directory for trained agent models.
- `tests/`: Unit and integration tests for the game logic and agents (e.g., `tests_rules.py`).
- `train/`: Scripts and utilities for training agents (e.g., `train_agent.py`, `train_self_play.py`).
- `truco/`: Core game logic, environment definition, and rules (`env.py`, `game_logic.py`, `rules.py`).
- `main.py`: Main entry point for running the application or playing the game.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/truco-zero.git
    cd truco-zero
    ```

2.  **Install `uv` (if you haven't already):**
    ```bash
    pip install uv
    # Or for a standalone installation:
    # curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    *Note: `uv` is a fast Python package installer and resolver. Using it is highly recommended.*

3.  **Use uv to sync**
    ```bash
    uv sync
    ```
    *Note: `uv` creates the virtual environment in a `.venv` directory by default.*

## Usage

### Running the Game / Playing against an Agent
To run the main game or play against a specific agent:
```bash
python main.py
```
*(You might need to adjust `main.py` or provide command-line arguments to select agents or game modes.)*

### Training an Agent
To train an agent using self-play:
```bash
python train/train_self_play.py
```
To train a specific agent:
```bash
python train/train_agent.py
```
*(Consult the scripts in the `train/` directory for available options and configurations.)*

### Evaluating Agents
To evaluate agents against each other or a human player:
```bash
python eval/player_vs_agent.py
python eval/player_vs_player.py
python eval/player_vs_random.py
```
*(Check the scripts in `eval/` for details on how to configure evaluations.)*


## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.

## License
This project is licensed under the [MIT License](LICENSE).

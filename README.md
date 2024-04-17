# TicTacToe_Agent

## Overview

This repository contains an implementation of a Tic Tac Toe environment and a Q-learning agent designed to play the game. The agent learns to play Tic Tac Toe through reinforcement learning, utilizing a neural network to approximate Q-values for state-action pairs.

## Files

- **main.py**: Orchestrates the training process of the Q-learning agent in the Tic Tac Toe environment. It configures logging, initializes the environment and agent, conducts training episodes, and saves the trained model.

- **q_learning_agent.py**: Implements the Q-learning agent, which learns to play Tic Tac Toe by interacting with the environment. The agent uses a neural network model to approximate Q-values and employs techniques like experience replay and epsilon-greedy exploration to learn an optimal policy.

- **tic_tac_toe_environment.py**: Contains the implementation of the Tic Tac Toe environment. The environment provides methods to initialize the game, take actions, and check for a winner. Rewards are defined for different game outcomes, and the environment keeps track of the game state.

- **pu_info.py**: Provides utilities to gather system information related to the GPU, including cuDNN version, NVIDIA driver information, and CUDA toolkit information. It also includes functions to extract CPU information.

## Requirements

- Python 3.9 - 3.11

## GPU Requirements (not mandatory)

To utilize GPU acceleration, ensure the following components are installed:

- TensorFlow version 2.15.0 or later
- cuDNN version 8.9
- CUDA version 12.2
- NVIDIA driver version 525.60.13 or later

For further details and additional information, please refer to the following resources:
 - [https://www.tensorflow.org/install/source#gpu](https://www.tensorflow.org/install/source#gpu)
 - [https://docs.nvidia.com/deploy/cuda-compatibility/index.html#default-to-minor-version](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#default-to-minor-version)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Eclipse91/TicTacToe_Agent.git
   ```

2. Navigate to the project directory:
   ```bash
   cd TicTacToe_Agent
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:

   ```bash
   python3 main.py
   ```

## Usage

1. **Setup**: Adjust parameters such as batch size, number of episodes, and model file paths to customize the training process. (1000 episodes takes around 20 minutes).

2. **Training**: Run the `main.py` script to start training the Q-learning agent in the Tic Tac Toe environment.

3. **Monitoring**: Monitor the training progress through the logged information and check the trained model after training completes.

4. **Testing**: At the end, you can test the model by playing as many games as you want against it.

## License

This project is licensed under the MIT LICENSE - see the [LICENSE](LICENSE) file for details.

## Notes

Feel free to contribute or report issues!
This README provides a clearer structure, concise information, and instructions for setting up and running the TicTacToe_Agent. Adjust the content as needed for your project.


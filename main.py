import os
import sys
import shutil
import logging
from time import time
from datetime import datetime
from numpy import count_nonzero
import tensorflow as tf
from pu_info import PUInfo
from q_learning_agent import QLearningAgent
from tic_tac_toe_environment import TicTacToeEnvironment


# Constants
BATCH_SIZE = 32
EPISODES = 1000
# If USE_GPU is True, the program will attempt to utilize the GPU. If it fails, a RuntimeError will be raised.
USE_GPU = False
# If MODEL_FILE_PATH is not empty, the program will attempt to load an existing model from that path. 
# Example './models/model_20240418_090434/model_20240418_090434.h5'
MODEL_FILE_PATH = ''
# If NEW_MODEL_FILE is not empty, the program will replace the existing model with the new one. It will be stored in ./models/. 
# Example 'my_model'
NEW_MODEL_FILE = ''


def box_text(text):
    '''
    Print a text in a box format.
    Args:
        text (str): The text to be printed in the box.
    Returns:
        log_box (str): The text boxed using *.
    '''
    log_box = ''
    lines = text.split('\n')
    width = max(len(line) for line in lines)
    log_box+=('*' * (width + 4)) + '\n'
    for line in lines:
        log_box += (f'* {line.ljust(width)} *' + '\n')
    log_box += ('*' * (width + 4)) + '\n'

    return log_box

def log_layer_values(agent):
    '''
    Logs the weights of each layer in the provided agent's model.
    Args:
        agent: An object containing a TensorFlow model with layers.

    '''
    logging.info(f'\nModel:')
    model = ''
    for i, layer in enumerate(agent.model.layers):
        model += f'Layer {i+1}\n'#{layer.get_weights()}\n'
        for weight in layer.get_weights():
            model += f'{weight}\n'
    logging.info(box_text(model))

def log_layer_info(layer):
    '''
    Log information about a Keras layer.
    Args:
        layer (tf.keras.layers.Layer): The Keras layer to log information about.
    Returns:
        layer_logs: Info about the Keras layer.
    '''
    layer_logs = ''
    layer_logs += f'Layer Name: {layer.name}\n'
    layer_logs += f'Layer Type: {layer.__class__.__name__}\n'
    try:
        layer_logs += f'Input Shape: {layer.input_dtype}\n'
    except:
        pass
    try:
        layer_logs += f'Output Shape: {layer.output_shape}\n'
    except:
        pass
    layer_logs += f'Number of Trainable Parameters: {layer.count_params()}\n'

    if hasattr(layer, 'activation'):
        layer_logs += f'Activation Function: {layer.activation.__name__}\n'

    if hasattr(layer, 'kernel_regularizer'):
        layer_logs += f'Kernel Regularizer: {layer.kernel_regularizer}\n'

    if hasattr(layer, 'bias_regularizer'):
        layer_logs += f'Bias Regularizer: {layer.bias_regularizer}\n'

    if hasattr(layer, 'get_weights'):
        weights = layer.get_weights()
        if len(weights) > 0:
            layer_logs += 'Weights:\n'
            for i, weight in enumerate(weights):
                layer_logs += f'  Weight {i+1}: {weight.shape}\n'
    layer_logs += f'Trainable: {layer.trainable}\n'

    if hasattr(layer, 'kernel_initializer'):
        layer_logs += f'Kernel Initializer: {layer.kernel_initializer.__class__.__name__}\n'

    if hasattr(layer, 'bias_initializer'):
        layer_logs += f'Bias Initializer: {layer.bias_initializer.__class__.__name__}\n'

    return layer_logs

def log_agent_info(agent, log_file):
    '''
    Log information about the Q-learning agent.
    Args:
        agent (QLearningAgent): The Q-learning agent to log information about.
        log_file (str): The path to the log file.
    '''
    # Parameters
    other_logs = (f'{"="*10}PARAMETERS{"="*10}\nBatch_size: {BATCH_SIZE}\nEpisodes: {EPISODES}\nUse_GPU: {USE_GPU}\n' 
                + (f'Old_model: {MODEL_FILE_PATH}\n' if MODEL_FILE_PATH else '')
                + (f'New_model: {NEW_MODEL_FILE}\n' if NEW_MODEL_FILE else ''))
   
    logging.info(box_text(other_logs))

    # Agent info
    agent_logs = f'{"="*10}AGENT PARAMETERS{"="*10}\n'

    attributes = vars(agent)
    for attr, value in attributes.items():
        agent_logs += f'{attr}: {value}\n'
    agent_logs += f''

    logging.info(box_text(agent_logs))

    # Open a file in write mode (creates the file if it doesn't exist, otherwise truncates it)
    with open(log_file, 'a') as f:
        # Redirect the stdout to the file
        sys.stdout = f
        # Print the summary of the model
        agent.model.summary()
        # Reset stdout to its original value
        sys.stdout = sys.__stdout__

    model_logs =f'{"="*10}MODEL{"="*10}\n'
    for i, layer in enumerate(agent.model.layers):
        model_logs += f'\nLayer {i+1} \n\n'
        model_logs += log_layer_info(layer)

    logging.info(box_text(model_logs))

    log_layer_values(agent)

def log_configurator():
    '''
    Configure and initialize the logger.
    '''
    log_directory = './logs/'
    os.makedirs(log_directory, exist_ok=True)
    current_datetime = datetime.now()
    current_file_name = os.path.splitext(os.path.basename(__file__))[0]
    formatted_datetime = current_datetime.strftime('%Y%m%d_%H%M%S')
    log_file = f'{log_directory}{current_file_name}_{formatted_datetime}.log'

    # logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')
    logging.info('Program started')

    return log_file

def extract_main_part(file_path):
    '''
    Extracts the main part of the file name without the extension and the directory.
    Args:
        file_path (str): The path of the file.

    Returns:
        str: The main part of the file name.
    '''
    file_name = os.path.basename(file_path)
    main_part = file_name.split('.')[0]
    return main_part

def create_folder_and_copy_files(folder_name):
    '''
    Creates a folder with the same name as the associated log file into the files folder.
    This utility can be useful for organizing experiments until you identify the best fitting model.    
    Args:
        folder_name (str): The name of the folder to be created.
    '''
    # Create the folder if it doesn't exist
    files_directory = './files/'
    os.makedirs(files_directory, exist_ok=True)
    folder_name = files_directory + extract_main_part(folder_name)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created successfully.')

    # Copy files into the folder
    shutil.copy('main.py', folder_name)
    shutil.copy('q_learning_agent.py', folder_name)
    shutil.copy('tic_tac_toe_environment.py', folder_name)

def save_model(agent, folder_name):
    '''
    Saves the model of the agent into the models folder in .h5 and .tflite format.
    Args:
        agent: The agent whose model needs to be saved.
        folder_name (str): The name of the folder to save the model into.
    '''
    models_directory = './models/'
    os.makedirs(models_directory, exist_ok=True)
    folder_name = models_directory + extract_main_part(folder_name).replace('main', 'model')
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created successfully.')

    file_path = folder_name + '/' + extract_main_part(folder_name)
    agent.save_model(file_path + '.h5')
    agent.save_model(file_path)
    
    converter = tf.lite.TFLiteConverter.from_saved_model(file_path)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model to a file
    with open(file_path + '.tflite', 'wb') as f:
        f.write(tflite_model)

def convert_board(board):
    '''
    Convert the numerical representation of the Tic-Tac-Toe board into a human-readable format.
    Args:
        board (list): The current state of the Tic-Tac-Toe board, represented as a list of integers.
    Returns:
        list: A human-readable representation of the Tic-Tac-Toe board, with 'X' representing Player 1's moves,
              'O' representing Player 2's moves, and numbers representing empty spaces.
    '''
    for i, value in enumerate(board):
        if value == 1:
            board[i] = 'X'
        elif value == 2:
            board[i] = 'O'
        else:
            board[i] = str(i)

    return board

def play_against_agent(env, agent):
    '''
    Play a game of Tic-Tac-Toe against the provided agent.
    Args:
        env (TicTacToeEnvironment): The TicTacToeEnvironment object representing the game environment.
        agent (QLearningAgent): The QLearningAgent object representing the AI agent to play against.
    '''
    while True:
        # Test the agent
        state = env._reset()
        state = state.flatten()
        done = False
        
        while not done:
            env.moves = 0
            board = state.tolist()
            board = convert_board(board)
            os.system('cls||clear')
            print(f'{board[:3]}\n{board[3:6]}\n{board[-3:]}\n')

            if count_nonzero(state == 0.0) % 2:
                action = agent.act(state)
            else:
                try:
                    action = int(input('Choose a number: '))
                except ValueError:
                    os.system('cls||clear')
                    print('Choose a number between 0 and 8')
            
            # Perform the chosen action and observe the next state and reward
            next_state, reward, done = env._step(action)
            next_state = next_state.flatten()

            # Store the experience tuple in the agent's memory
            state = next_state

            winner = env.check_winner()
        if winner:
            board = state.tolist()
            board = convert_board(board)
            os.system('cls||clear')
            print(f'{board[:3]}\n{board[3:6]}\n{board[-3:]}\n')
            print(f'Player {winner} won')
        elif winner == None:
            board = state.tolist()
            board = convert_board(board)
            os.system('cls||clear')
            print(f'{board[:3]}\n{board[3:6]}\n{board[-3:]}\n')
            print(f"It's a DRAW")
        
        if input('Do you want to play again? [y/n] ') != 'y':
            break

def main():
    # Record the start time
    start = time()

    # Configure and initialize the logger file
    log_file = log_configurator()

    # Copy the files to memorize the configuration
    create_folder_and_copy_files(log_file)
    
    # Log GPU or CPU info
    if USE_GPU:
        logging.info(box_text(PUInfo.log_gpu_info()))
    else:
        # Disable CUDA devices visibility to TensorFlow by setting CUDA_VISIBLE_DEVICES to an empty string
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        logging.info(box_text(PUInfo.log_cpu_info()))

    # Initialize the environment and agent
    env = TicTacToeEnvironment()
    agent = QLearningAgent(state_size=9, action_size=9, path_to_model=MODEL_FILE_PATH)

    # Add agent info
    log_agent_info(agent, log_file)

    # Loop through episodes
    for episode in range(EPISODES):
        # Record start time
        start_time = time()

        # Reset the environment and get the initial state
        state = env._reset()
        state = state.flatten()
        done = False

        # Loop through steps within the episode until termination
        while not done:
            # Choose an action using the agent's policy
            action = agent.act(state)

            # Perform the chosen action and observe the next state and reward
            next_state, reward, done = env._step(action)
            next_state = next_state.flatten()
    
            # Store the experience tuple in the agent's memory
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            # Perform experience replay if the agent's memory is sufficiently filled
            if len(agent.memory) > BATCH_SIZE:
                agent.replay(BATCH_SIZE)

            logging.info(f'Action: {action}, Reward: {reward}, Done: {done}, Episode: {episode}, Epsilon: {agent.epsilon}')

        # Record end time
        end_time = time()

        # Calculate duration
        duration = end_time - start_time
        logging.info(f'Episode duration: {duration} seconds')

    # Save the trained model
    if NEW_MODEL_FILE:
        save_model(agent, NEW_MODEL_FILE)
    else:
        save_model(agent, log_file)

    # Log the trained model
    log_layer_values(agent)

    # Calculate and log the total duration in both seconds and minutes with seconds formatted to two decimal places
    duration = time() - start
    minutes = int(duration // 60)  # Get whole minutes
    seconds = duration % 60  # Get remaining seconds
    logging.info(f'Total duration: {duration} seconds')
    logging.info(f'Total duration: {minutes} minutes and {seconds:.2f} seconds')  # Format seconds with 2 decimal places

    play_against_agent(env, agent)

    
if __name__ == '__main__':
    main()

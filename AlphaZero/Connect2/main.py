# import torch

from AlphaZero.Connect2.game import Connect2Game
from AlphaZero.Connect2.model_tf import Connect2Model
from AlphaZero.Connect2.trainer_tf import Trainer
from tensorflow.keras.models import load_model

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args = {
    'batch_size': 64,
    'numIters': 30,                                # Total number of training iterations
    'numEps': 3,                                  # Number of full games (episodes) to run during each iteration
    'num_simulations': 10,                         # MCTS simulations
    'temperature': 0.2,                             # temperature for "exploration"
    # 'numItersForTrainExamplesHistory': 20,
    'epochs': 2,                                    # Number of epochs of training per iteration
    'checkpoint_path': 'latest.pth'                 # location to save latest set of weights

}

game = Connect2Game()
board_size = game.get_board_size()
action_size = game.get_action_size()

model = load_model('connect2.h5')
# model = Connect2Model(board_size, action_size)

trainer = Trainer(game, args)
model = trainer.learn(model, lr=1E-3, verbose=1)
model.save('connect2.h5')

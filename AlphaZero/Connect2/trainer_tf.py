import numpy as np
from time import time
from monte_carlo_tree_search import MCTS
from tensorflow.keras import optimizers as tfo
import matplotlib.pyplot as plt

class Trainer:

    def __init__(self, game, args):
        self.game = game
        self.args = args
        self.mcts = None  # MCTS(self.game, self.model, self.args)

    def exceute_episode(self, model):
        train_examples = []
        current_player = 1
        episode_step = 0
        state = self.game.get_init_board()

        while True:
            episode_step += 1

            canonical_board = self.game.get_canonical_board(state, current_player)

            self.mcts = MCTS(self.game, self.args)
            root = self.mcts.run(model, canonical_board, to_play=1)

            action_probs = list(np.zeros((self.game.get_action_size(), ), dtype=int))
            for k, v in root.children.items():
                action_probs[k] = v.visit_count

            action_probs = action_probs / np.sum(action_probs)
            train_examples.append((canonical_board, current_player, action_probs))

            action = root.select_action(temperature=self.args['temperature'])
            state, current_player = self.game.get_next_state(state, current_player, action)
            reward = self.game.get_reward_for_player(state, current_player)

            if reward is not None:
                ret = []
                for hist_state, hist_current_player, hist_action_probs in train_examples:
                    # [Board, currentPlayer, actionProbabilities, Reward]
                    ret.append((hist_state, hist_action_probs, reward * ((-1) ** (hist_current_player != current_player))))

                return ret

    def learn(self, model, lr=1e-3, verbose=1):
        model.compile(loss=["mse", "mse"], optimizer=tfo.Adam(lr))
        loss = []
        txt_fmt = "%0" + str(len(str(self.args['numIters']))) + "d"
        if verbose == 2:
            plt.figure();
        for i in range(1, self.args['numIters'] + 1):
            tic = time()
            train_examples = []

            for eps in range(self.args['numEps']):
                iteration_train_examples = self.exceute_episode(model)
                train_examples.extend(iteration_train_examples)

            in_state = np.array([i[0] for i in train_examples])
            out_act = np.array([i[1] for i in train_examples])
            out_score = np.array([i[2] for i in train_examples])

            I = np.random.choice(out_score.size, out_score.size)
            model.fit(in_state[I], [out_act[I], out_score[I]], batch_size=self.args['batch_size'],
                           epochs=self.args['epochs'], verbose=0)
            res = model.evaluate(in_state[I], [out_act[I], out_score[I]], verbose=0)
            toc = time() - tic
            text = txt_fmt + "/" + txt_fmt + " (%3.1fs)- action_loss: %1.5f - value_loss: %1.5f"
            text = text % (i, self.args['numIters'], toc, res[1], res[2])
            if verbose == 1:
                print(text)
            if verbose == 2:
                loss.append(np.array(res[1:])[None])
                plt.cla()
                plt.plot(np.row_stack(loss));
                plt.xlabel('Games')
                plt.legend(['Action', 'Value'])
                plt.title(text)
                plt.grid(True)
                plt.pause(.01)

        return model


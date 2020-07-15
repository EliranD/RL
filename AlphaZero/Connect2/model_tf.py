from tensorflow.keras import layers as tfl
from tensorflow.keras import models as tfm


def Connect2Model(board_size, action_size):
    inp = tfl.Input(shape=(board_size,))
    x = tfl.Dense(16, activation='linear')(inp)
    x = tfl.Dense(16, activation='linear')(x)
    val = tfl.Dense(1, activation='linear', name='value')(x)  # the decision scaore
    action = tfl.Dense(action_size, activation='softmax', name='action')(x)  # the decision scaore
    model = tfm.Model(inputs=inp, outputs=[action, val])
    return model

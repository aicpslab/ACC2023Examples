import torch
import torch.nn as nn
import numpy as np


def generate_2d_batch_data(model: nn.Module, inputs: np.ndarray):

    batch_size = len(inputs[0])
    # Empty Numpy Array in proper shape
    outputs = np.empty((0, 2))

    for batch in inputs:
        output = model(torch.tensor(batch).to("cpu"))

        # Add the output to the array
        outputs = np.vstack((outputs, output.detach().numpy()))

        np.reshape(outputs, (len(inputs), batch_size, 2))

    return outputs


def generate_1d_batch_data(model: nn.Module, inputs: np.ndarray):

    batch_size = len(inputs[0])
    # Empty Numpy Array in proper shape
    outputs = np.empty((0, 1))

    for batch in inputs:

        output = model(torch.tensor(batch))
        outputs = np.vstack((outputs, output.detach().numpy()))

    outputs = np.reshape(outputs, (len(inputs), batch_size, 1))

    return outputs

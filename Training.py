import torch
import torch.nn as nn
import numpy as np


class Training:

    @staticmethod
    def train(model: nn.Module, inputs: np.ndarray, expected_outputs: np.ndarray, loss_fn, optimizer, device, epochs: int):

        # For every Epoch
        for i in range(epochs):
            print("-----------------------------------")
            print(f"Epoch: {i + 1}")

            # Train this epoch
            Training._train_one_epoch(model, inputs, expected_outputs, loss_fn, optimizer, device)

    def _train_one_epoch(model, inputs, expected_outputs, loss_fn, optimizer, device):

        # For every input and it's expected value
        for input, expected in zip(inputs, expected_outputs):

            # Create Tensors for PyTorch
            input = torch.tensor(input).to(device)
            expected = torch.tensor(expected).to(device)

            # Make an inference
            prediction = model(input)

            # Update the loss based on the inference
            loss = loss_fn(prediction, expected)

            # Update the Network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":

    import numpy as np

    torch.manual_seed(2003)
    np.random.seed(2003)

    from DataSetClasses import OneDimensionalData
    from OutputDataGeneration import generate_1d_batch_data

    from Models.LargeNetworkPytorch import LargeNetwork
    from Models.PrunedNetworkPytorch import PrunedNetwork

    # Instantiate the Large and Pruned Network classes
    LargeModel = LargeNetwork()
    PrunedModel = PrunedNetwork()

    # Generate One Dimensional Data using an interval of [-2,2],
    # 10_000 inputs, and a batch size of 10.
    TrainingData = OneDimensionalData([-2, 2], 10_000, 10)

    epochs = 50

    learning_rate = 0.01
    loss_fn = nn.L1Loss() # MAE, Mean Absolute Error
    optimizer = torch.optim.Adam(PrunedModel.parameters(), lr=learning_rate)

    LargeNetworkOutputData = generate_1d_batch_data(LargeModel, TrainingData.GetBatches())

    # Train the Network
    Training.train(PrunedModel, TrainingData.GetBatches(), LargeNetworkOutputData, loss_fn, optimizer, "cpu", epochs)

    # Set the Model to Evaluation Mode
    PrunedModel.eval()

    torch.save(LargeModel, "Models\\TrainedModels\\LargeModel.pt")
    torch.save(PrunedModel, "Models\\TrainedModels\\PrunedModel.pt")

import torch.nn as nn


class PrunedNetwork(nn.Module):
    def __init__(self):
        super(PrunedNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),

            nn.Linear(10, 10),
            nn.ReLU(),

            nn.Linear(10, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            module.bias.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        x = self.layers(x)
        return x

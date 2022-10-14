# This File will generate a merged neural network
# if given a larger network and a smalelr network.
# The output of this network is the difference in the
# outputs of the larger and smaller network.

# 8/19/2022
# Wesley Cooke

# Dependencies
import torch
import torch.nn as nn
import copy as cp
import numpy as np


class GenerateMergedNet:

    @staticmethod
    def from_PyTorch(largePath, smallPath):
        LargeModel = torch.load(largePath)
        SmallModel = torch.load(smallPath)

        LargeModelDict = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
        SmallModelDict = GenerateMergedNet._extract_pytorch_wandb(SmallModel)

        return GenerateMergedNet.from_WandB(LargeModelDict, SmallModelDict)

    @staticmethod
    def _extract_quant_pytorch(model):
        pass

    @staticmethod
    def _extract_pytorch_wandb(model):

        weights = []
        biases = []
        weights_shape = []
        biases_shape = []
        actFuncs = []

        for name, param in model.named_parameters():

            if name.endswith(".weight"):
                weights.append(cp.deepcopy(param.data.numpy()))
                weights_shape.append(tuple(param.data.shape))
            elif name.endswith(".bias"):
                temp = np.expand_dims(cp.deepcopy(param.data.numpy()), axis=1)
                biases.append(temp)
                biases_shape.append(tuple(biases[-1].shape))

        for param in model.modules():
            if isinstance(param, nn.ReLU):
                actFuncs.append("ReLU")
            elif isinstance(param, nn.Sigmoid):
                actFuncs.append("sigmoid")
            elif isinstance(param, nn.Tanh):
                actFuncs.append("tanh")

        # actFuncs.append(actFuncs[-1])
            # :TODO: Reconize More Activation functions

        model_dict = {
            "w": weights,
            "b": biases,
            "w_shapes": weights_shape,
            "b_shapes": biases_shape,
            "acts": actFuncs
        }

        return model_dict

    @staticmethod 
    def from_TensorFlow(largePath, smallPath):
        # :TODO: Implement generating a merged network
        # from TensorFlow files.

        raise NotImplementedError

    @staticmethod
    def from_onnx(largePath, smallPath):
        # :TODO: Implement generating a merged network
        # from Onnx files.

        raise NotImplementedError

    @staticmethod
    def from_WandB(aLargeModelDict, aSmallModelDict):
        # Dictionary must contain weights, biases, activation functions,
        # the shape of the weights, and the shape of the biases.

        # :TODO: Implement generating a merged network 
        # from a dictionary that contains the following:
        # w: weights, b: biases, w_shapes: weights_shape, b_shapes: biases_shape, acts: activation Functiosn

        # Unpack Dictionaries 
        wLar = aLargeModelDict['w']
        bLar = aLargeModelDict['b']
        wLarShape = aLargeModelDict['w_shapes']
        bLarShape = aLargeModelDict['b_shapes']
        aLar = aLargeModelDict['acts']

        wSmall = aSmallModelDict['w']
        bSmall = aSmallModelDict['b']
        wSmallShape = aSmallModelDict['w_shapes']
        bSmallShape = aSmallModelDict['b_shapes']
        aSmall = aSmallModelDict['acts']

        numLayersLar = len(wLar)
        numLayersSmall = len(wSmall)

        # Merged Model Variables
        wMerged = []
        bMerged = []
        # Assume the Inner Activation Functions
        # are the same and pass up information without change
        aMerged = aLar

        # Layer One
        wMerged.append(np.vstack((wLar[0], wSmall[0])))
        bMerged.append(np.concatenate((bLar[0], bSmall[0])))

        # Layer 2 - Hidden Layers up to the output layer of the small model
        for i in range(1, numLayersSmall-1):  # Range function does < 2nd argument. NOT <= second. IE. i never takes the value of numLaySmall-1

            tempWL = np.hstack((wLar[i], np.zeros((wLarShape[i][0], wSmallShape[i][0]))))
            tempWS = np.hstack((np.zeros((bSmallShape[i][0], bLarShape[i][0])), wSmall[i]))

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], bSmall[i])))

        # Expanded Layers for the small model
        # If the Small Model and Large Model have the same layers, this will be skipped
        for i in range(numLayersSmall-1, numLayersLar-1):
            tempWL = np.hstack((wLar[i], np.zeros((wLarShape[i][0], wSmallShape[numLayersSmall-2][0]))))
            tempWS = np.hstack((np.zeros((wSmallShape[numLayersSmall-2][0], wLarShape[i][0])), np.eye(wSmallShape[numLayersSmall-2][0]))) # eye is identity matrix

            wMerged.append(np.concatenate((tempWL, tempWS)))
            bMerged.append(np.concatenate((bLar[i], np.expand_dims(np.zeros(wSmallShape[numLayersSmall-2][0]), axis=1))))

        # Parallel output layer for the Small and Large Model. 
        tempWL = np.hstack((wLar[numLayersLar-1], np.zeros((wLarShape[numLayersLar-1][0], wSmallShape[-1][1]))))
        tempWS = np.hstack((np.zeros((wSmallShape[-1][0], wLarShape[-1][1])), wSmall[numLayersSmall-1]))

        wMerged.append(np.concatenate((tempWL, tempWS)))
        bMerged.append(np.concatenate((bLar[numLayersLar-1], bSmall[numLayersSmall-1])))

        # Final output layer that will compute the difference of the two outputs
        # Final Layer
        wMerged.append(np.hstack((np.eye(wLarShape[-1][0]), -np.eye(wLarShape[-1][0]))))
        bMerged.append(np.zeros((2*bLarShape[-1][1], 1)))

        for index, bias in enumerate(bMerged):
            bMerged[index] = np.squeeze(bias)

        mergedDict = {
            "w": wMerged,
            "b": bMerged,
            "acts": aMerged
        }

        return mergedDict

    @staticmethod
    def truncate_params(ModelDict, num):
        """
        Take the weights and biases and truncate them to num decimal places.

        Returns a new Model dictionary
        """

        q_w = []
        for weight in ModelDict['w']:
            q_w.append(np.around(weight, num).astype(np.float16))

        q_b = []
        for bias in ModelDict['b']:
            q_b.append(np.around(bias, num).astype(np.float16))

        QuantDict = {
            "w": q_w,
            "b": q_b,
            "w_shapes": ModelDict['w_shapes'],
            "b_shapes": ModelDict["b_shapes"],
            "acts": ModelDict["acts"]
        }

        return QuantDict

    @staticmethod
    def generated_sequential_from_dictionary(ModelDict):

        w = ModelDict['w']
        w_shapes = ModelDict['w_shapes']
        b = ModelDict['b']
        b_shapes = ModelDict['b_shapes']
        activations = ModelDict['acts']

        seq = nn.Sequential()

        # count the number of linear layers we add.
        num_linear = 0
        index = 0

        for layer in activations:
            in_features, out_features = tuple(reversed(w_shapes[index]))

            seq.append(nn.Linear(in_features, out_features))
            num_linear += 1

            if layer == "ReLU":
                seq.append(nn.ReLU())
            else:
                return NotImplementedError
            index += 1

        # Last Layer
        in_features, out_features = tuple(reversed(w_shapes[-1]))
        seq.append(nn.Linear(in_features, out_features)) # Output layers of the two separate networks
        index += 1
        num_linear += 1

        # Has to be true or the model will be wrong
        assert num_linear == len(w)

        # Init weights and biases from merged net
        index = 0
        for layer in seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.nn.parameter.Parameter(torch.tensor(w[index]))
                layer.bias.data = torch.nn.parameter.Parameter(torch.tensor(b[index]))
                index += 1

        # Return the Sequential model
        return seq

    @staticmethod
    def generate_merged_sequential(ModelDict):

        weights = ModelDict['w']
        biases = ModelDict['b']
        activations = ModelDict['acts']

        seq = nn.Sequential()

        # count the number of linear layers we add.
        num_linear = 0
        index = 0
        for layer in activations:
            seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0]))
            num_linear += 1

            if layer == "ReLU":
                seq.append(nn.ReLU())
            else:
                return NotImplementedError
            index += 1

        # Last two linear layers for mergeed network
        seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0])) # Output layers of the two separate networks
        index += 1
        num_linear += 1
        seq.append(nn.Linear(weights[index].shape[-1], weights[index].shape[0])) # Output of the merged network: IE The difference in the two separate networks
        num_linear += 1

        # Has to be true or the model will be wrong
        assert num_linear == len(weights)

        # Init weights and biases from merged net
        index = 0
        for layer in seq:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.nn.parameter.Parameter(torch.tensor(weights[index]))
                layer.bias.data = torch.nn.parameter.Parameter(torch.tensor(biases[index]))
                index += 1

        # Return the Sequential model
        return seq

    @staticmethod
    def simulate_relu_network(ModelDict, inputs):

        num_inf = len(inputs)
        inputs = np.reshape(inputs, (num_inf, 1, 1))

        outputs = []

        for input in inputs:
            for weight, bias in zip(ModelDict["w"][:-1], ModelDict['b'][:-1]):

                input = np.matmul(weight, input) + bias
                input = np.maximum(0, input)

            output = np.matmul(ModelDict['w'][-1], input) + ModelDict['b'][-1]
            outputs.append(output)

        outputs = np.reshape(outputs, (1, num_inf)).squeeze()

        return outputs

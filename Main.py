# Pip Imports used
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import freeze_support

# Custom Imports Used
from GenerateMerged import GenerateMergedNet
from DataSetClasses import OneDimensionalData
from Models.LargeNetworkPytorch import LargeNetwork
from Models.PrunedNetworkPytorch import PrunedNetwork

# Veritex Tools Used
from veritex.utils.plot_poly import plot_polytope2d
from veritex.utils.sfproperty import Property
from veritex.networks.ffnn import FFNN
from veritex.sets.vzono import VzonoFFNN
from veritex.methods.reachplot import run
from veritex.methods.shared import SharedState
from veritex.methods.worker import Worker


def get_max_from_verticies(exact_output_sets):

    max_distance = 0

    # Veritex returns multiple lists of verticies due to parallel processing.

    for a_set in exact_output_sets:
        for a_vertice in a_set:
            distance = np.linalg.norm(a_vertice)
            max_distance = max(distance, max_distance)

    return max_distance


def get_exact_output_sets(SequentialModel, lbs, ubs):

    # Code to get the exact output sets from Veritex. Can be seen in their examples.

    # Veritex only uses the Sequential part of the model.
    dnn = FFNN(SequentialModel, exact_outputd=True)

    # Define the input set using the lower and upper bounds
    # We don't specify an unsafe input domain.
    property_1 = Property([lbs, ubs], [], set_type='FVIM')

    # Setting up the Parallel Processing Framework
    processes = []
    num_processors = mp.cpu_count()
    shared_state = SharedState(property_1, num_processors)
    one_worker = Worker(dnn)

    # Starting the Parallel Processing
    for index in range(num_processors):
        p = mp.Process(target=one_worker.main_func, args=(index, shared_state))
        processes.append(p)
        p.start()

    # Waiting for Process to finish
    for p in processes:
        p.join()

    # Gathering Output Sets
    outputs = []
    while not shared_state.outputs.empty():
        outputs.append(shared_state.outputs.get())

    # Extract vertices of output reachable sets
    exact_output_sets = [np.dot(item.vertices, item.M.T) + item.b.T for item in outputs]

    return exact_output_sets


def plot_difference(LargeDictionary, SmallDictionary, max_distance, legend, file):

    # Plot the data
    input_data = OneDimensionalData([0, 1], 20_000, 10)
    inputs = input_data.GetData()
    LargeOutputs = GenerateMergedNet.simulate_relu_network(LargeDictionary, inputs)
    SmallOutputs = GenerateMergedNet.simulate_relu_network(SmallDictionary, inputs)

    plt.xlabel("Inputs")
    plt.ylabel("Outputs")
    plt.rc('font', family='serif')

    smallLine = plt.plot(inputs, SmallOutputs, "r.", markersize=1)
    largeLine = plt.plot(inputs, LargeOutputs, "b.",  markersize=1)
    upperBound = plt.plot(inputs, LargeOutputs + max_distance, "g.",  markersize=1)
    lowerBound = plt.plot(inputs, LargeOutputs - max_distance, "c.",  markersize=1)

    plt.legend(legend)

    plt.show()
    plt.savefig(file)

if __name__ == '__main__':
    freeze_support()

    # Load the LargeModel, Quantize it, and generate the Merged Network
    LargePath = "Models\\TrainedModels\\LargeModel.pt"
    LargeModel = torch.load(LargePath)
    LargeDictionary = GenerateMergedNet._extract_pytorch_wandb(LargeModel)
    LargeQuantizedDictionary = GenerateMergedNet.truncate_params(LargeDictionary, 4)
    MergedLargeQuant = GenerateMergedNet.from_WandB(LargeDictionary, LargeQuantizedDictionary)
    SeqLargeQuant = GenerateMergedNet.generate_merged_sequential(MergedLargeQuant)

    # Get the exact Output Sets
    lbs, ubs = [0], [1]
    exact_output_sets = get_exact_output_sets(SeqLargeQuant, lbs, ubs)
    max_distance = get_max_from_verticies(exact_output_sets)

    # Display Data
    print("Max Distance of Large and LargeQuant: ", max_distance)
    legend = ["Quantized Model", "Large Model", "UpperBound", "Lower Bound"]
    file = "Figs\LargeLargeQuant_Distance.png"
    plot_difference(LargeDictionary, LargeQuantizedDictionary, max_distance, legend, file)

    # Load the PrunedModel, Quantize it, and generate another Merged Network
    PrunedPath = "Models\\TrainedModels\\PrunedModel.pt"
    PrunedModel = torch.load(PrunedPath)
    PrunedDictionary = GenerateMergedNet._extract_pytorch_wandb(PrunedModel)
    PrunedQuantizedDictionary = GenerateMergedNet.truncate_params(PrunedDictionary, 4)
    MergedLargePrunedQuant = GenerateMergedNet.from_WandB(LargeDictionary, PrunedQuantizedDictionary)
    SeqLargePruned = GenerateMergedNet.generate_merged_sequential(MergedLargePrunedQuant)

    exact_output_sets = get_exact_output_sets(SeqLargePruned, lbs, ubs)
    max_distance = get_max_from_verticies(exact_output_sets)

    print("Max Distance of Large and PrunedQuant: ", max_distance)
    legend = ["Pruned and Quantized Model", "Large Model", "UpperBound", "Lower Bound"]
    file = "Figs\LargePrunedQuant.png"
    plot_difference(LargeDictionary, PrunedQuantizedDictionary, max_distance, legend, file)

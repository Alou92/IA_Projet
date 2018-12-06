from src.DataManager_td import DataManager
from src.NeuralNetwork_td import NeuralNetwork
import numpy as np


def main():
    #loss: 15.552 , acc: 0.0967

    manager = DataManager()
	
    manager.preprocessData()
	
    network = NeuralNetwork()
	
    network.createModel()
	
    network.train(manager.datagen,manager.train_data,manager.train_labels,10)
	
    results = network.evaluate(manager.eval_data,manager.eval_labels)

if __name__ == "__main__":
    main()
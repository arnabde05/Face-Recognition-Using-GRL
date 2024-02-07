# Face-Recognition-Using-GRL

# Face Recognition using Graph Representation Learning

This repository contains code and resources for performing face recognition using Graph Representation Learning. We have implemented three different methods for this purpose: Node2Vec, DeepWalk, and GraRep.

## Table of Contents

- [Introduction](#introduction)
- [Methods](#methods)
  - [Node2Vec](#node2vec)
  - [DeepWalk](#deepwalk)
  - [GraRep](#grarep)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

In the field of face recognition, Graph Representation Learning has shown promising results. This repository explores three different methods for face recognition using graphs: Node2Vec, DeepWalk, and GraRep. Each method has its unique approach to learning representations from graph data.

## Methods

### Node2Vec

Node2Vec is a graph embedding technique that learns continuous representations for nodes in a graph. In the context of face recognition, Node2Vec can be used to capture meaningful features from face graphs.

The node2vec framework learns low-dimensional representations for nodes in a graph by optimizing a neighborhood preserving objective. The objective is flexible, and the algorithm accomodates for various definitions of network neighborhoods by simulating biased random walks. Specifically, it provides a way of balancing the exploration-exploitation tradeoff that in turn leads to representations obeying a spectrum of equivalences from homophily to structural equivalence.

![image](https://github.com/arnabde05/Face-Recognition-Using-GRL/assets/87455060/bcd27060-c174-415a-8213-46474eae4568)

After transitioning to node v from t, the return hyperparameter, p and the inout hyperparameter, q control the probability of a walk staying inward revisiting nodes (t), staying close to the preceeding nodes (x1), or moving outward farther away (x2, x3).

![image](https://github.com/arnabde05/Face-Recognition-Using-GRL/assets/87455060/49c878ac-718d-4c66-bb35-7d850df81835)



### DeepWalk

DeepWalk is another graph embedding method that leverages random walks to generate embeddings for nodes. DeepWalk can be applied to face graphs to create feature representations for face recognition tasks.

### GraRep

GraRep stands for Graph Representation Learning via Graph Partitioning. It is a method that considers higher-order proximity information in graphs, which can be beneficial for capturing complex relationships in face data.

## Getting Started

To get started with this project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Download the necessary datasets or prepare your own face graph data.
4. Choose the method (Node2Vec, DeepWalk, or GraRep) you want to use and navigate to the respective directory for detailed instructions.

## Usage

Provide clear instructions on how to use your code and apply the three different methods for face recognition. Include examples, command-line options, and any relevant configuration details.

## Contributing

We welcome contributions from the community. If you have improvements, bug fixes, or new features to add, please follow our [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the [MIT License](LICENSE).

Here is the complete workflow of the project. 

![Proposed model (poster) _changed](https://github.com/arnabde05/Face-Recognition-Using-GRL/assets/87455060/96c85bde-70fc-44eb-bf38-d7a8c75ba26c)


I just gave a short overview of all techniques I used. Actually this is the first approach I tried. so, for more informations and everything please contact with me. 


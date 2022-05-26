# Implicit Hypergraph Neural Networks

This repository contains my thesis or project work for the `Mathematics expert in data analytics and machine learning
Postgraduate specialist programme` of ELTE TTK.
I have also created a C language program, an earlier version of which is published here.
The topic of my thesis is implicit neural networks.
In a network or a graph representation of any complex system, cycles or loops correspond to feedback, which is
an important cornerstone of their operation. Such systems can be analyzed in the framework of the implicit
neural networks since the output of the nodes during the training procedure cannot be given explicitly.

My original aims were:
* to creating a program to simulate neural networks in which the (hyper)graphs are given by the user in a very flexible way.
* get to know the C language.

**Log**
* 11/07/21 Read in the inputs in Python from file and create the inputs for the C code.
* 30/07/21 Create the base C code and parallelised by OpenMP.
* 17/08/21 Rearrange the folder structure.
* 16/09/21 Implementing the PERT method for optimizing the calculations in feedforward networks.
* 29/09/21 Implementing the mutability of the weights.
* 02/10/21 Random sampling for the training set.
* 18/10/21 RAdam implementation.
* 14/11/21 Time series: Encoder-Decoder (not published yet).
* 30/12/21 Implementing multiclass and multilabel classification.
* 03/02/22 Important fixes.
* 16/02/22 BCE with logits.
* 25/03/22 Memory management correctly for minibatches, predictions for datasets without labels, saving the weights of the optimizers.
* 26/05/22 Important fixes related to integer types.

**In progress**
* 26/05/22 Most of the GPU kernels are ready (not published yet)

**TODO**
* Convolutional networks (the C code is ready, python scripts to building convnets)
* Sequential version
* Applying the results for loseless compression
* 
* Python wrappers
* Documentation webpage
* Tutorials

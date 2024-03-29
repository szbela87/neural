# Implicit Neural Networks

This repository contains my thesis work for the `Mathematics Expert in Data Analytics and Machine Learning
postgraduate specialization program` of ELTE TTK (Budapest, Hungary).
I have also created a C language program, a demo version of this is published here.
The topic of my thesis is implicit neural networks.
In a network or a graph representation of any complex system, cycles or loops correspond to feedback, which is
an important cornerstone of their operation. Such systems can be analyzed in the framework of the implicit
neural networks since the output of the nodes during the training procedure cannot be given explicitly.
We investigate also an even more general implicit structure, where a neuron can have multiple inputs.
Here, after summing the signals at the inputs and activating them separately, the neuron activation value
is the product of them. Note that convolutional networks are special cases of this architecture, and are also
building blocks of LSTM networks. Our results are summarized in the pdf file in the `doc` directory which contains
a short user manual for the program as well.

**My original aims were:**
* to creating a program to simulate neural networks 
* where the (hyper)graphs which represent the network are given by the user in a very flexible way.
* get to know the C and Cuda C programming languages.

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
* 31/05/22 CPU code for validating the GPU code
* 15/06/22 GPU code

**TODO**
* Convolutional networks (the cpu-version can handle weight sharing) by python scripts
* Sequential architectures (i.e. LSTM, GRU)
* Applying the results for lossless compression
* Writing Python wrappers
* Creating Documentation webpage
* Creating Tutorials

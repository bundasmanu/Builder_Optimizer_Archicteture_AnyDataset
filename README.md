# Archicteture to build and optimize image dataset's

## Concept

This repository represents a backbone that can be adopted and adapted to solve any kind of image classification problem.
It provides a set of "tools" that are already pre-configured and can be easily adapted to solve any image classification problem. This allows to reduce the work needed to solve the problems.

## Project Setup
To use this repository, the users should follow these steps:
1.  According to the needs, users can clone the project, or pull all its content in an existing project:
    * If users want to clone the repository, just write: git clone https://github.com/bundasmanu/Builder_Optimizer_Archicteture_AnyDataset.git
    * If users want to pull: git pull https://github.com/bundasmanu/Builder_Optimizer_Archicteture_AnyDataset.git master --allow-unrelated-histories
2.  Install requirements: pip install -r requirements.txt;
3.  If needed you should adapt the config.py file, which includes configuration variables for algorithms, architectures, etc. 

<!--## Class Diagram (base):
[Architecture Image](breast_Class_Diagram.png)-->

## Features of Project
- It was built using the best practices of oriented object pattern, such as: object factories, strategy, template, among others;
- It provides pre-configured skeletons of several convolutional architectures: AlexNet, VGGNet, ResNet and DenseNet;
- Users can easily adapt networks with different structures, different values of their hyperparameters, training approaches, etc;
- The whole process of building, training, testing and providing the results of the models is developed and is completely autonomous;
- In addition, some techniques are also available to improve the generalization of the models, and to reduce some constraints, such as: low number of samples, unbalanced classes. The techniques included are: Random undersampling, Random oversampling and Data Augmentation. More techniques can be easily created and adapted to project.
- Finally, it is also provided the possibility for the user to optimize his models, using optimization algorithms such as Particle Swarm Optimization (PSO), or Genetic Algorithm (GA).
- A standard optimization flow is provided. It allows the user to quickly manage the optimization of convolutional networks, allowing an easy definition of the limits of the problem, of the objective function, of the iterative cycle of particles or the presentation of graphics.

## Examples of usage
- Resolution of Breast Cancer Dataset: https://github.com/bundasmanu/breast_histopathology
- Resolution of Skin Mnist Dataset: https://github.com/bundasmanu/skin_mnist
- Resolution of Colorectal Cancer Dataset: https://github.com/bundasmanu/Colorectal_Histopathology

## Licence
GPL-3.0 License

I am open to new ideas and improvements to the current repository. However, until the defense of my master thesis, I will not accept pull request's.

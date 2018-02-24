# Decision Tree Implementations


## Introduction


This module includes different implementation of decision tree:
* The first implementation uses scikit-learn framework.
* The second one is a decision tree written in pure python and build using the CART algorithm.
* The third one is a decision tree written in pure python and build using the ID3 algorithm.


## Motivation

The purpose of this exercice is to gain a better understanding on how decision tree are built. Decision Tree are still very popular because of their simplicity and interpretability.

There are different families of algorithms to build decision tree:
* ID3
* C4.5
* C5.0
* CART (Classification and Regression Trees)

They mainly differ from "which questions to ask and when". (https://www.youtube.com/watch?v=LDRbO9a6XPU)

It is also a valuable exercice to practice programming skills. 2 important python concepts are used in these implementations : classes and objects (decision node, leaf and question) for modeling the tree and recursion for building it.

## Code structure

The code is structured as follow :

```
pyDecisionTree
│
├- data/
│   └- us_election/
│
├- library/
│   ├- python_CART/
│   ├- python_ID3/
│   └- scikit_learn/
│
├- performance/
│   └- num_bench/
│
├- unittest/
│   └- test_core.py
│
├- evaluate.py
├- predict.py
├- prepare.py
├- train.py
│
├- .gitignore
├- README.md
└- requirements.txt
```


The models are in the folder **library**.


## Installation

To use the different implementations, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyDecisionTree.git
```

### Using a virtual environment

First create the virtual environment :

```
$ python3 -m venv path_to_the_env
```

Activate it :

```
$ source path_to_the_env/bin/activate
```

Then install all the requirements :

```
$ pip install -r requirements.txt
```

## Test

To test if all the functionnalities are working :

```
$ python -m unittest discover -s unittest
```


## Author

Rémi Moise

moise.remi@gmail.com

## License

MIT License

Copyright (c) 2017
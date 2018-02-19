# Decision Tree Implementations

**Which questions to ask and when?**

## Introduction

There are many types algorithms to build decision tree:
* ID3
* C4.5
* C5.0
* CART (Classification and Regression Trees)

They differ from "which questions to ask and when". (https://www.youtube.com/watch?v=LDRbO9a6XPU)

This module presents 2 ways of implementing a decision tree :
* The first implementation uses scikit-learn framework.
* The second one is coded from scratch.

The implemented model is a decision tree classifier with the following attributes :
* blabla
* blabla
* blabla


## Motivation

The purposes of this exercice are to :
* gain a better understanding on how does decision tree work.
* practice python programming.
* practice scikit-learn framework implementation.

## Code structure

The code is structured as follow :

```
pyLogisticRegression
│
├- data/
│   └- us_election/
│
├- library/
│   ├- diy/
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
├- docs/
│   └- Lect09.pdf
│
├- .gitignore
├- README.md
└- requirements.txt
```

## Installation

To use the different implementations, you can directly clone the repository :

```
$ git clone https://github.com/lamsremi/pyDecisionTreeID3.git
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
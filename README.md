# nqs-tensorflow (Neural-Network Quantum States implementation in Tensorflow)

This project is an implementation of a small part of neural-network quantum states in Tensorflow to speed-up the process with graphics processing units (GPU). The implementation is based on the [NetKet library](https://www.netket.org/index.html) [1] and Science paper by Carleo and Troyer [2]. We also propose several transfer learning protocol for the scalability of the neural-network quantum states based on our paper here [3].

## Requirements
This project is based on Python programming language. We suggest  to use Python 2 instead of Python 3.
These are the library requirements for the project:
* `tensorflow==1.15` or `tensorflow-gpu==1.15`
* `numpy`
* `scipy`
* `matplotlib`

It is also available as requirements.txt in the project and do
``pip install -r requirements.txt``
to install the necessary libraries.

## Usage
We have provided some scripts named ``script-[model].py`` as an example to run the program with the given parameters. The description of the scripts are the following:
* ``script-ising.py``: run one-dimensional Ising model from cold-start.
* ``script-ising-transfer.py``: run one-dimensional Ising model with transfer.
* ``script-heisenberg.py``: run one-dimensional Heisenberg model from cold-start.
* ``script-heisenberg-2d.py``: run two-dimensional Heisenberg model from cold-start.
* ``script-heisenberg-transfer.py``: run one-dimensional Heisenberg model with transfer.
* ``script-heisenberg-2d-transfer.py``: run two-dimensional Heisenberg model with transfer.

The parameters for the script are explained in ``script-ising.py``. To run, simply use the command ``python script-ising.py``. The script will create a folder called `results` to store all the results (this can be changed in the script). 


## References
[1] G.  Carleo,   K.  Choo,   D.  Hofmann,   J.  E.  T.  Smith,T.  Westerhout,  F.  Alet,  E.  J.  Davis,  S.  Efthymiou,I. Glasser, S.-H. Lin, M. Mauri, G. Mazzola, C. B. Mendl,E. van Nieuwenburg, O. O’Reilly, H. Th ́eveniaut, G. Tor-lai,  F.  Vicentini,  and  A.  Wietek,  SoftwareX ,  100311(2019).

[2] G. Carleo and M. Troyer, Science 355, 602 (2017)

[3]  R.  Zen,  L.  My,  R.  Tan,  F.  Hebert,  M.  Gattobigio,C. Miniatura, D. Poletti,  and S. Bressan, arXiv:1908.09883  (2019).

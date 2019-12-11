from __future__ import print_function
import os
import tensorflow as tf
from functools import partial
import pickle
import numpy as np
from machine.rbm.real import RBMReal
from hamiltonian import Ising, Heisenberg
from graph import Hypercube
from sampler import Gibbs, MetropolisExchange, MetropolisLocal
from learner import Learner
from logger import Logger
from observable import * 

# System
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#np.random.seed(123)
#tf.set_random_seed(123)


for iteration in range(5):
    ### Parameters for the graph of the model
    # length of the lattice
    lattice_length = 16
    # dimension
    dimension = 1
    # periodic boundary condition (True) or open boundary condition (False)
    pbc = False

    ### Parameters for the Hamiltonian
    # Type of the Hamiltonian
    hamiltonian_type = "ISING"
    #hamiltonian_type = "HEISENBERG"
    # Parameters of the Hamiltonian
    h = 1.0
    jx = 1.0
    jy = 1.0
    jz = 2.0

    ### Parameters for the Sampler
    # Number of samples
    num_samples = 10000
    # Number of steps in the sampling process
    num_steps = 1

    ### Parameters for the RBM
    # Density (ratio between the number of hidden and visible nodes)
    density = 2
    # Function to initialise the weight
    initializer = partial(np.random.normal, loc= 0.0, scale=0.01)

    ### Parameters for the Learner
    # Initialise tensorflow session
    sess = tf.Session()
    # Optimiser for the gradient descent
    trainer = tf.train.RMSPropOptimizer
    # Initial learning of the optimiser
    learning_rate = 0.001
    # The number of iterations/epochs for the training
    num_epochs = 10000
    window_period = 200
    # Size of the minibatch 
    minibatch_size = 0 
    # Threshold value for the stopping criterion
    stopping_threshold = 0.005
    # Initialise reference energy
    reference_energy = None
    # If you want to compare with DMRG value
    use_dmrg_reference = True

    ### Parameters for the Logger
    log = True
    # The path for the result folder
    result_path = './results/'
    # The name of the subpath for your experiment, by default if it is empty it will be named 'cold-start' for cold start
    subpath = ''
    # Indicate whether you want to visualise the weight or visible layer and how frequent
    visualize_weight = False
    visualize_visible = False
    visualize_freq = 10
    # The list of observables that you wish to compute
    observables = [MagnetizationZ, MagnetizationZSquareFerro, MagnetizationZSquareAntiFerro, CorrelationZ]
    # Indicate whether you want to see the weight different after and before training
    weight_diff = True

    #### Create instances from all of the parameters
    graph = Hypercube(lattice_length, dimension, pbc)

    hamiltonian = None
    if hamiltonian_type == "ISING":
        hamiltonian = Ising(graph, jz, h)
    elif hamiltonian_type == "HEISENBERG":
        hamiltonian = Heisenberg(graph, jx, jy, jz)

    ## Choose the type of sampler here
    sampler = Gibbs(num_samples, num_steps)
    #sampler = MetropolisExchange(num_samples, num_steps)
    #sampler = MetropolisLocal(num_samples, num_steps)
    machine = RBMReal(graph.num_points, density, initializer, num_expe=iteration, use_bias=False)
    machine.create_variable()

    if use_dmrg_reference:
        if hamiltonian_type == "ISING": 
            refs = pickle.load(open('ising-energy-dmrg.p', 'r'))
        elif hamiltonian_type == "HEISENBERG":
            refs = pickle.load(open('heisenberg-dmrg-energy.p', 'r'))
        if lattice_length in refs:
            if jz in refs[lattice_length]:
                reference_energy = float(refs[lattice_length][jz])
                print('True energy:', reference_energy) 

    learner = Learner(sess, graph, hamiltonian, machine, sampler, trainer, learning_rate, num_epochs, minibatch_size,
                      window_period, reference_energy, stopping_threshold, visualize_weight, visualize_visible, visualize_freq)

    logger = Logger(log, result_path, subpath, visualize_weight, visualize_visible, visualize_freq, observables, weight_diff)

    learner.learn()
    logger.log(learner)

    # clear previous graph for multiple runs of learner
    tf.reset_default_graph()

    sess.close()

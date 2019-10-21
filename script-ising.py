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
    # Graph
    lattice_length = 16
    dimension = 1
    pbc = False

    # Hamiltonian
    hamiltonian_type = "ISING"
    #hamiltonian_type = "HEISENBERG"
    h = 1.0
    jx = 1.0
    jy = 1.0
    jz = 2.0

    # Sampler
    num_samples = 10000
    num_steps = 1

    # Machine config
    # Rbm
    density = 2
    initializer = partial(np.random.normal, loc= 0.0, scale=0.01)

    # Learner
    sess = tf.Session()
    trainer = tf.train.RMSPropOptimizer
    learning_rate = 0.001
    num_epochs = 10000
    window_period = 200
    minibatch_size = 0 
    stopping_threshold = 0.005
    reference_energy = None
    use_dmrg_reference = True

    # Logger
    log = True
    result_path = './results/'
    subpath = ''
    visualize_weight = False
    visualize_visible = False
    visualize_freq = 10
    observables = [MagnetizationZ, MagnetizationZSquareFerro, MagnetizationZSquareAntiFerro, CorrelationZ]
    weight_diff = True

    # create instances
    graph = Hypercube(lattice_length, dimension, pbc)

    hamiltonian = None
    if hamiltonian_type == "ISING":
        hamiltonian = Ising(graph, jz, h)
    elif hamiltonian_type == "HEISENBERG":
        hamiltonian = Heisenberg(graph, jx, jy, jz)

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

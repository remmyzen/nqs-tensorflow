from __future__ import print_function
import os
import tensorflow as tf
from functools import partial
import pickle
import numpy as np
from machine.rbm.real import RBMReal
from machine.rbm import RBMTransfer
from hamiltonian import Ising, Heisenberg
from graph import Hypercube
from sampler import Gibbs, MetropolisExchange, MetropolisLocal
from learner import Learner
from logger import Logger
from observable import * 

# System
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#np.random.seed(123)
#tf.set_random_seed(123)


for iteration in range(5):
    # Graph
    lattice_length = 8
    dimension = 1
    pbc = False
    if pbc:
        pbc_str = 'pbc'
    else:
        pbc_str = 'obc'

    # Hamiltonian
    #hamiltonian_type = "ISING"
    hamiltonian_type = "HEISENBERG"
    h = 1.0
    jx = 1.0
    jy = 1.0
    jz = -2.0

    # Sampler
    num_samples = 10000
    num_steps = 1000

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

    # transfer (k,p)-tiling, p is defined automatically
    k_val = 1  # (1,2)-tiling
    #k_val = 2 # (2,2)-tiling
    #k_val = lattice_length / 2  # (L,p)-tiling 

    # Logger
    log = True
    result_path = './results/'
    subpath = '%d,p_tiling' % k_val
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

    #sampler = Gibbs(num_samples, num_steps)
    sampler = MetropolisExchange(num_samples, num_steps)
    #sampler = MetropolisLocal(num_samples, num_steps)
    machine = RBMReal(graph.num_points, density, initializer, num_expe=iteration, use_bias=False)

    if hamiltonian_type == "ISING":
        if lattice_length == 8:
            transfer = RBMTransfer(machine, graph, '%sising_%dd_%d_%d_%.2f_1.00_%s/cold-start/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str), iteration)
        else:
            transfer = RBMTransfer(machine, graph, '%sising_%dd_%d_%d_%.2f_1.00_%s/%s/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str, subpath), iteration)
    elif hamiltonian_type == "HEISENBERG":
        if lattice_length == 8:
            transfer = RBMTransfer(machine, graph, '%sheisenberg_%dd_%d_%d_1.00_1.00_%.2f_%s/cold-start/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str), iteration)
        else:
            transfer = RBMTransfer(machine, graph, '%sheisenberg_%dd_%d_%d_1.00_1.00_%.2f_%s/%s/' % (result_path, dimension, lattice_length / 2, density, jz, pbc_str, subpath), iteration)

    transfer.tiling(k_val)

    machine.create_variable()


    if use_dmrg_reference:
        if hamiltonian_type == "ISING": 
            refs = pickle.load(open('ising-energy-dmrg.p', 'r'))
        elif hamiltonian_type == "HEISENBERG":
            refs = pickle.load(open('heisenberg-energy-dmrg.p', 'r'))
        if lattice_length in refs:
            if jz in refs[lattice_length]:
                reference_energy = float(refs[lattice_length][jz])
                print('True energy:', reference_energy) 

    learner = Learner(sess, graph, hamiltonian, machine, sampler, trainer, learning_rate, num_epochs, minibatch_size,
                      window_period, reference_energy, stopping_threshold, visualize_weight, visualize_visible, visualize_freq)

    logger = Logger(log, result_path, subpath, visualize_weight, visualize_visible, visualize_freq, observables, weight_diff)

    learner.learn()
    logger.log(learner)

    logger.visualize_weights(transfer.W_base, logger.result_path, 0, 'before transfer', transfer.learner_base)
    logger.visualize_weights(transfer.W_transfer, logger.result_path, 1, 'after transfer', learner)

    # clear previous graph for multiple runs of learner
    tf.reset_default_graph()

    sess.close()

from __future__ import print_function
import tensorflow as tf
import time
import numpy as np
import copy
import scipy.stats


class Learner:

    def __init__(self, session, graph, hamiltonian, machine, sampler, trainer, learning_rate, num_epochs=1000,
                 minibatch_size=0, window_period=100, reference_energy=None, stopping_threshold=0.005,
                 visualize_weight=False, visualize_visible=False, visualize_freq=10, observables=[], observable_freq = 0):
        self.graph = graph
        self.hamiltonian = hamiltonian
        self.machine = machine
        self.sampler = sampler
        self.trainer = trainer
        self.learning_rate = learning_rate
        self.session = session
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.window_period = window_period
        self.reference_energy = reference_energy
        self.stopping_threshold = stopping_threshold
        self.observables = observables
        self.observable_freq = observable_freq

        self.ground_energy = []
        self.ground_energy_std = []
        self.ground_energy_skew = []
        self.ground_energy_kurtosis = []
        self.ground_energy_quantile = []
        self.energy_burns = []
        self.energy_burns_std = []
        self.rel_errors = []
        self.times = []
        self.all_energies = []
        self.observables_value = []

        self.rbm_weights = []
        self.rbm_visible_bias = []
        self.rbm_hidden_bias = []
        self.probs_all = []
        self.sample_set = []
        self.sample_set_final = []
        self.visualize_weight = visualize_weight
        self.visualize_visible = visualize_visible
        self.visualize_freq = visualize_freq

        if self.minibatch_size == 0 or self.minibatch_size > self.sampler.num_samples:
            self.minibatch_size = self.sampler.num_samples

    def learn(self):
        sample_set_np = self.sampler.get_initial_random_samples(self.machine.num_visible)
        sample_set = tf.placeholder(tf.float32, [None, self.machine.num_visible], name="input_x")
        eloc = self.get_energy(sample_set)
        grads = self.get_gradient(sample_set, self.minibatch_size, eloc)
        new_sample = self.sampler.sample(self.machine, sample_set, self.minibatch_size)

        # variables used for mini-batch update
        num_minibatch = self.sampler.num_samples / self.minibatch_size
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tv.initialized_value(), trainable=False) for tv in tvs]
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        accum_ops = [accum_vars[i].assign_add(grad) for i, grad in enumerate(grads)]
        apply_ops = self.trainer(self.learning_rate).apply_gradients([(accum_vars[i] / num_minibatch, tv) for i, tv in enumerate(tf.trainable_variables())])

        self.reset_memory_array()

        self.session.run(tf.global_variables_initializer())

        energy_burn = 0.0
        energy_burn_std = 0.0

        # Main learning loop
        for epoch in range(self.num_epochs):
            start = time.time()

            # initialize variables for mini-batch update
            self.session.run(zero_ops)
            ground_energy_acc = []
            ground_energy_std_arr = []
            ground_energy_skew_arr = []
            ground_energy_kurtosis_arr = []
            ground_energy_quantile_arr = []
            new_sample_set_np = []

            # mini-batch update loop
            for i in range(num_minibatch):
                batch_sample_set = sample_set_np[i * self.minibatch_size:(i + 1) * self.minibatch_size]
                feed_dict_ = {sample_set: batch_sample_set}
                eloc_np, o, mini_new_sample_np = self.session.run([eloc, accum_ops, new_sample], feed_dict=feed_dict_)
                ground_energy_acc.append(np.mean(eloc_np))
                ground_energy_std_arr.append(np.std(eloc_np))
                ground_energy_skew_arr.append(scipy.stats.skew(eloc_np))
                ground_energy_kurtosis_arr.append(scipy.stats.kurtosis(eloc_np))
                ground_energy_quantile_arr.append([np.quantile(eloc_np, 0), np.quantile(eloc_np, 0.25), np.quantile(eloc_np, 0.5),np.quantile(eloc_np, 0.75), np.quantile(eloc_np, 1)])
                new_sample_set_np.append(mini_new_sample_np)

            # process energy and error
            ground_energy, ground_energy_std, energy_burn, energy_burn_std, rel_error = self.process_energy_and_error(epoch, ground_energy_acc,
                                                                       ground_energy_std_arr, ground_energy_skew_arr, ground_energy_kurtosis_arr,
                                                                       ground_energy_quantile_arr)

            # visualize
            self.visualize(epoch, sample_set_np)

            # update parameters
            self.session.run(apply_ops)

            end = time.time()
            time_interval = end - start
            self.times.append(time_interval)

            # update sample set
            sample_set_np = np.array(new_sample_set_np).reshape(self.sampler.num_samples, self.machine.num_visible)
            self.sample_set = sample_set_np

            # calculate observable
            if self.observable_freq != 0 and epoch % self.observable_freq == 0:
                obs = self.calculate_observables(sample_set_np)            

            print('Epoch: %d, energy: %.4f, std: %.4f, std / mean: %.4f, relerror: %.5f, time: %.4f' % (
            epoch, ground_energy, ground_energy_std, ground_energy_std / np.abs(ground_energy), rel_error, time_interval))

            ### stop if it is NaN (fail)
            if np.isnan(ground_energy):
                print('Fail NaN')
                last_epoch = 99999
                break

            # check stopping criteria
            if ground_energy_std / np.abs(ground_energy) < self.stopping_threshold:
                break

        # save the last data
        self.visualize(epoch, sample_set_np, last=True)

        return None

    def calculate_observables(self, sample_set_np): 
        confs, count_ = np.unique(sample_set_np, axis=0, return_counts=True)
        prob_out = count_ / float(self.sampler.num_samples)
        value_map = {}
        temp_map = {}
        for obs in self.observables:
            observ = obs(prob_out, confs, self.machine.num_visible)
            obs_value = observ.get_value()
            value_map[observ.get_name()] = obs_value 
            temp_map[observ.get_name()] = observ.get_accumulate_value(obs_value)
            
        self.observables_value.append(value_map)
        return temp_map
        

    def get_energy(self, sample_set):
        hamiltonian = self.hamiltonian.calculate_hamiltonian_matrix(sample_set, self.minibatch_size)
        lvd = self.hamiltonian.calculate_lvd(sample_set, self.machine, self.minibatch_size)

        eloc_array = tf.reduce_sum((tf.exp(0.5 * lvd) * hamiltonian), axis=1, keepdims=True)

        return eloc_array

    def get_gradient(self, sample_set, sample_size, eloc):
        derlogs = self.machine.derlog(sample_set, sample_size)
        grad_dict = {}

        eloc_mean = tf.reduce_mean(eloc, axis=0, keepdims=True)
        for name, derlog in derlogs.items():
            derlog_mean = tf.reduce_mean(derlog, axis=0, keepdims=True)
            grad_dict[name] = 2 * tf.matmul(tf.transpose(eloc - eloc_mean), tf.conj(derlog - derlog_mean)) / self.minibatch_size   

        grads = self.machine.reshape_grads(grad_dict)
        return grads

    def reset_memory_array(self):
        self.ground_energy = []
        self.ground_energy_std = []
        self.ground_energy_skew = []
        self.ground_energy_kurtosis = []
        self.ground_energy_quantile = []
        self.energy_burns = []
        self.energy_burns_std = []
        self.rel_errors = []
        self.times = []
        self.all_energies = []
        self.sample_set = []
        self.sample_set_final = []
        self.observables_value = []

    def visualize(self, epoch, sample_set_np, last=False):
        if last or epoch == 0:
            self.rbm_weights.append((epoch, '', self.session.run(self.machine.get_parameters()[0])))
            self.rbm_visible_bias.append((epoch, '', self.session.run(self.machine.get_parameters()[1])))
            self.rbm_hidden_bias.append((epoch, '', self.session.run(self.machine.get_parameters()[2])))
        else:
            if self.visualize_weight and epoch % self.visualize_freq == 0:
                self.rbm_weights.append((epoch, '', self.session.run(self.machine.get_parameters()[0])))
                self.rbm_visible_bias.append((epoch, '', self.session.run(self.machine.get_parameters()[1])))
                self.rbm_hidden_bias.append((epoch, '', self.session.run(self.machine.get_parameters()[2])))
            if self.visualize_visible and epoch % self.visualize_freq == 0:
                vprob = self.session.run(self.sampler.get_vprob(sample_set_np, self.machine))
                self.probs_all.append((epoch, '', np.average(vprob, axis=0)))

    def process_energy_and_error(self, epoch, ground_energy_acc, ground_energy_std_arr, ground_energy_skew_arr, ground_energy_kurtosis_arr, ground_energy_quantile_arr):
        ground_energy = np.real(np.mean(ground_energy_acc))
        self.ground_energy.append(ground_energy)
        ground_energy_std = np.real(np.mean(ground_energy_std_arr))
        self.ground_energy_std.append(ground_energy_std)
        self.ground_energy_skew.append(np.real(np.mean(ground_energy_skew_arr)))
        self.ground_energy_kurtosis.append(np.real(np.mean(ground_energy_kurtosis_arr)))
        self.ground_energy_quantile.append(np.real(np.mean(ground_energy_quantile_arr, 0)))
        energy_burn = np.mean(self.ground_energy[-self.window_period:])
        energy_burn_std = np.std(self.ground_energy[-self.window_period:])
        self.energy_burns.append(energy_burn)
        self.energy_burns_std.append(energy_burn_std)
        if self.reference_energy is None:
            rel_error = 0
        else:
            rel_error = np.abs((ground_energy - self.reference_energy) / self.reference_energy)
        self.rel_errors.append(rel_error)
        return ground_energy, ground_energy_std, energy_burn, energy_burn_std, rel_error

    def make_pickle_object(self):
        temp_learner = copy.copy(self)
        temp_learner.machine = temp_learner.machine.make_pickle_object(self.session)
        temp_learner.session = None
        return temp_learner

    def to_xml(self):
        str = ""
        str += "<learner>\n"
        str += "\t<params>\n"
        str += "\t\t<optimizer>%s</optimizer>\n" % self.trainer
        str += "\t\t<lr>%.5f</lr>\n" % self.learning_rate
        str += "\t\t<epochs>%d</epochs>\n" % self.num_epochs
        str += "\t\t<minibatch>%d</minibatch>\n" % self.minibatch_size
        str += "\t\t<window_period>%d</window_period>\n" % self.window_period
        str += "\t\t<stopping_threshold>%d</stopping_threshold>\n" % self.stopping_threshold
        str += "\t</params>\n"
        str += "</learner>\n"
        return str

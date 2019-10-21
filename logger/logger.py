from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Logger(object):
    default_result_path = './'
    default_subpath = 'cold-start'

    def __init__(self, log=True, result_path=default_result_path, subpath=default_subpath, visualize_weight=False,
                 visualize_visible=False, visualize_freq=10, observables=None, weight_diff=False):
        self.log_value = log
        self.result_path = result_path
        self.subpath = subpath
        if self.subpath is None or self.subpath == '':
            self.subpath = Logger.default_subpath
        self.subpath = '/' + self.subpath + '/'
        self.visualize_weight = visualize_weight
        self.visualize_visible = visualize_visible
        self.visualize_freq = visualize_freq
        self.observables = observables
        self.weight_diff = weight_diff

    def log(self, learner):
        if self.log_value is True:
            model_name = self.get_model_name(learner)
            self.make_base_path(model_name)
            self.print_model(learner)
            self.visualize_energy(learner)
            self.write_logs(learner)
            self.visualize_params(learner)
            if self.visualize_visible:
                self.visualize_visible_rbm(learner)
            if self.observables is not None and len(self.observables) > 0:
                self.calculate_observables(learner) 
            if self.weight_diff:
                self.calculate_weight_difference(learner)

            self.save_model(learner)

    def get_model_name(self, learner):
        hamiltonian_name = learner.hamiltonian.__class__.__name__
        if learner.graph.pbc:
            bc = 'pbc'
        else:
            bc = 'obc'
        if hamiltonian_name == 'Heisenberg':
            model_name = 'heisenberg_%dd_%d_%d_%.2f_%.2f_%.2f_%s' % (
            learner.graph.dimension, learner.graph.length, learner.machine.density, learner.hamiltonian.jx,
            learner.hamiltonian.jy, learner.hamiltonian.jz, bc)

        elif hamiltonian_name == 'Ising':
            model_name = 'ising_%dd_%d_%d_%.2f_%.2f_%s' % (
            learner.graph.dimension, learner.graph.length, learner.machine.density, learner.hamiltonian.j,
            learner.hamiltonian.h, bc)
        else:
            model_name = 'unknown'
        return model_name

    def make_base_path(self, name):

        path = self.result_path + name + self.subpath
        self.make_directory(path)
        self.make_experiment_logs(path)

        # retrieve all subdirectory names
        dir_names = [int(f) for f in os.listdir(path) if os.path.isdir(path + f)]
        if len(dir_names) > 0:
            self.num_experiment = max(dir_names) + 1
        else:
            self.num_experiment = 0
        next_dir = str(self.num_experiment) + '/'
        path = path + next_dir
        self.make_directory(path)
        self.result_path = path

    def make_experiment_logs(self, path):
        if not os.path.exists(path + 'experiment_logs.csv'):
            with open(path + 'experiment_logs.csv', 'w') as f:
                f.write('num,ground_energy,ground_energy_std,variance,epoch,time\n')
                f.close()
        self.parent_result_path = path

    def make_directory(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_model(self, learner):
        filename = 'model.p'
        path = self.result_path + filename
        pickle.dump(learner.make_pickle_object(), open(path, 'wb'))
        print('====== Model saved in %s ======' % (path))

    def print_model(self, learner):
        filename = 'model_desc.xml'
        path = self.result_path + filename
        ff = open(path, 'w')
        ff.write('<model>\n')
        ff.write(learner.to_xml())
        ff.write(learner.graph.to_xml())
        ff.write(learner.hamiltonian.to_xml())
        ff.write(learner.machine.to_xml())
        ff.write(learner.sampler.to_xml())
        ff.write('</model>\n')
        ff.close()

    def calculate_observables(self, learner, num_samples=10000, num_steps=30, num_division=1):
        learner.sampler.set_num_samples(num_samples / num_division)

        if learner.sampler.__class__.__name__ == 'MetropolisExchange':
            num_steps = 5000
        elif learner.sampler.__class__.__name__ == 'Gibbs':
            num_steps = 500        
    
        learner.sampler.set_num_steps(num_steps)

        if len(learner.sample_set) == 0:
            params = np.array([])
            for i in range(num_division): 
                init_samples = learner.sampler.get_initial_random_samples(learner.machine.num_visible)
                samples = learner.session.run(learner.sampler.sample(learner.machine, init_samples, num_samples / num_division))
                if params.size == 0:
                    params = samples
                else:
                    params = np.concatenate((params,samples), 0)
        else:

            init_samples = learner.sample_set
            params = learner.session.run(learner.sampler.sample(learner.machine, init_samples, num_samples / num_division))
            learner.sample_set_final = params

            new_energy = learner.session.run(learner.get_energy(params))
            filename = 'energy.txt'
            ff = open(self.result_path + filename, 'w')
            ff.write('%.5f\n' % np.mean(new_energy))
            ff.write('%.5f\n' % np.std(new_energy))
            ff.close()


        confs, count_ = np.unique(params, axis=0, return_counts=True)
        prob_out = count_ / float(num_samples)
        if len(prob_out) < 5000:
            np.savetxt(self.result_path + 'probs.txt', prob_out)
            np.savetxt(self.result_path + 'confs.txt', confs)
            pickle.dump((confs, prob_out), open(self.result_path + 'probs.p', 'w'))
        for obs in self.observables:
            observ = obs(prob_out, confs, learner.machine.num_visible)
            obs_value = observ.get_value()
            filename = observ.get_name() + '.txt'
            ff = open(self.result_path + filename, 'w')
            ff.write(str(obs_value.tolist())) 
            ff.close()


    def visualize_energy(self, learner):
        plt.figure()
        plt.title('Energy vs Iteration, %s, %s' % (str(learner.hamiltonian), str(learner.machine)))
        plt.ylabel('Energy')
        plt.xlabel('Iteration #')
        ground_energy = np.array(learner.ground_energy)
        ground_energy_std = np.array(learner.ground_energy_std)
        plt.plot(range(len(ground_energy)), ground_energy, label='energy')
        plt.fill_between(range(len(ground_energy)), ground_energy - ground_energy_std,
                         ground_energy + ground_energy_std, alpha=0.4, color='red')
        if learner.reference_energy is not None:
            plt.axhline(y=learner.reference_energy, xmin=0, xmax=learner.num_epochs, linewidth=2, color='k',
                        label='Exact')

        plt.legend(frameon=False)
        plt.tight_layout()
        plt.show()
        plt.savefig(self.result_path + '/iter-energy.png')
        plt.close()

    def write_logs(self, learner):
        ground_energys = learner.ground_energy
        ground_energy_stds = learner.ground_energy_std
        ground_energy_skews = learner.ground_energy_skew
        ground_energy_kurtosises = learner.ground_energy_kurtosis
        energy_burns = learner.energy_burns
        energy_burns_std = learner.energy_burns_std
        rel_errors = learner.rel_errors
        times = learner.times[1:]

        filename = 'logs.csv'
        path = self.result_path + filename
        ff = open(path, 'w')
        ff.write('epoch,ground_energy,ground_energy_std,ground_energy_skew,ground_energy_kurtosis,ground_energy_window,rel_error,time\n')
        for ep, (ge, ges, gesk, gek, gew, gest, re, ti) in enumerate(
                zip(ground_energys, ground_energy_stds, ground_energy_skews, ground_energy_kurtosises, energy_burns, energy_burns_std, rel_errors, times)):
            ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (ep, ge, ges, gesk, gek, gew, gest, re, ti))
        ff.close()

        filename = 'experiment_logs.csv'
        path = self.parent_result_path + filename
        ff = open(path, 'a')
        ff.write('%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (
        self.num_experiment, energy_burns[-1], energy_burns_std[-1], ground_energy_stds[-1], len(ground_energys) - 1, np.sum(times), ground_energys[-1], ground_energy_stds[-1]))
        ff.close()

    def calculate_weight_difference(self, learner):
        filename = 'weight_diff.txt'
        ff = open(self.result_path + filename, 'w')
        _, _, first_weight = learner.rbm_weights[0]
        _, _, last_weight = learner.rbm_weights[-1]
        ff.write('%.5f' % ((first_weight.real - last_weight.real) ** 2).mean())
        ff.close()
         

    def visualize_params(self, learner):
        self.visualize_weights_rbm(learner)
        self.visualize_visible_bias_rbm(learner)
        self.visualize_hidden_bias_rbm(learner)

    def visualize_weights_rbm(self, learner):
        path = self.result_path + '/weights/'
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch, title, w in learner.rbm_weights:
            self.visualize_weights(w.real, path, epoch, title, learner)

    def visualize_visible_bias_rbm(self, learner):
        path = self.result_path + '/visible_bias/'
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch, title, bv in learner.rbm_visible_bias:
            self.visualize_bias(bv[0], path, epoch, 'visible', learner)

    def visualize_hidden_bias_rbm(self, learner):
        path = self.result_path + '/hidden_bias/'
        if not os.path.exists(path):
            os.makedirs(path)

        for epoch, title, bh in learner.rbm_hidden_bias:
            self.visualize_bias(bh[0], path, epoch, 'hidden', learner)

    def visualize_bias(self, bias, path, epoch, title, learner=None):
        plt.title('%s bias %05d, %s, %s' % (title, epoch, str(learner.hamiltonian), str(learner.machine)))
        plt.plot(bias)
        plt.xlabel('Bias')
        plt.ylabel('Value')
        plt.ylim(-1.3, 1.3)
        plt.tight_layout()
        plt.savefig(path + '/%s-bias-%05d.png' % (title, epoch))
        plt.close()

    def visualize_weights(self, weight, path, epoch=0, title='', learner=None):
        w = np.real(weight)
        plt.figure(figsize=(20, 10))
        plt.title('Weights %05d, %s, %s, %s' % (epoch, str(learner.hamiltonian), str(learner.machine), title))
        plt.imshow(w, cmap='hot', interpolation='nearest', vmin=-2.0, vmax=2.0)
        plt.xlabel('Hidden node')
        plt.ylabel('Visible node')
        plt.xticks(np.arange(0, w.shape[1], 1.0))
        plt.yticks(np.arange(0, w.shape[0], 1.0))
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path + '/weights-%05d.png' % epoch)
        plt.close()

    def visualize_visible_rbm(self, learner):
        path = self.result_path + '/visible/'
        if not os.path.exists(path):
            os.makedirs(path)

        for i, probs_all in enumerate(learner.probs_all):
            epoch = i * self.visualize_freq
            plt.figure(figsize=(20, 5))
            plt.title('Visible Layer Probability %s, %s, Epoch %05d' % (
            str(learner.hamiltonian), str(learner.machine), epoch))
            ax = plt.axes()
            for ii, b in enumerate(probs_all):
                ax.annotate('%.4f' % b,
                            xy=((ii * 2) + np.sin(b * np.pi), -np.cos(b * np.pi)), xycoords='data',
                            xytext=((ii * 2), 0), textcoords='data',
                            arrowprops=dict(arrowstyle="simple"), va='center', ha='center')
            ax.set_xlim(-1, len(probs_all) * 2)
            ax.set_ylim(-1, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.tight_layout()
            plt.savefig(path + '/vis-%05d.png' % epoch)
            plt.close()

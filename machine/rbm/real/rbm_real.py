from machine.rbm import RBM
import tensorflow as tf
import copy
from functools import partial
import numpy as np

class RBMReal(RBM):

    def __init__(self, num_visible, density=2, initializer=None, num_expe=None, use_bias=True):
        RBM.__init__(self, num_visible, density)
        self.initializer = initializer
        self.use_bias = use_bias
        self.num_expe = num_expe
        self.build_model()

    def build_model(self):
        self.random_initialize() 

    def random_initialize(self):
        if self.num_expe is not None:
            np.random.seed(self.num_expe)
        self.W_array = self.initializer(size=(self.num_visible, self.num_hidden)) 
        self.bv_array = np.zeros((1, self.num_visible))
        self.bh_array = np.zeros((1, self.num_hidden))

        np.random.seed()

    def create_variable(self):
        self.W = tf.Variable(tf.convert_to_tensor(value=self.W_array.astype(np.float32)), name="weights", trainable=True)
        self.bv = tf.Variable(tf.convert_to_tensor(value=self.bv_array.astype(np.float32)), name="visible_bias", trainable=True)
        self.bh = tf.Variable(tf.convert_to_tensor(value=self.bh_array.astype(np.float32)), name="hidden_bias", trainable=True)

    # Calculate log of p_RBM with configuration v
    def log_val(self, v):
        theta = tf.matmul(v, self.W) + self.bh
        sum_ln_thetas = tf.reduce_sum(input_tensor=tf.math.log(2*tf.cosh(theta)), axis=1, keepdims=True)
        ln_bias = tf.matmul(v, tf.transpose(a=self.bv))
        return sum_ln_thetas + ln_bias

    def log_val_diff(self, v1, v2):
        log_val_1 = self.log_val(v1)
        log_val_2 = self.log_val(v2)
        return log_val_1 - log_val_2

    def derlog(self, v, sample_size):
        theta = tf.matmul(v, self.W) + self.bh
        if self.use_bias:
            D_bv = v * 0.5
            D_bh = tf.tanh(theta) * 0.5
        else:
            D_bv = v * 0.0
            D_bh = tf.tanh(theta) * 0.0
        D_w_temp = tf.reshape(tf.tanh(theta), (sample_size, 1, self.num_hidden)) * tf.reshape(v, (sample_size, self.num_visible, 1)) * 0.5

        D_w = tf.reshape(D_w_temp, (sample_size, self.num_visible * self.num_hidden))

        derlog_dict = {'w': D_w, 'v': D_bv, 'h': D_bh}
        return derlog_dict

    def reshape_grads(self, grad_dict):
        grad_bv = tf.reshape(grad_dict['v'], (1, self.num_visible))
        grad_bh = tf.reshape(grad_dict['h'], (1, self.num_hidden))
        grad_w = tf.reshape(grad_dict['w'], (self.num_visible, self.num_hidden))
        return [grad_w, grad_bv, grad_bh]

    # helpers for sampling
    def get_new_visible(self, v):
        hprob = self.get_hidden_prob_given_visible(v)
        hstate = self.convert_from_prob_to_state(hprob)
        vprob = self.get_visible_prob_given_hidden(hstate)
        vstate = self.convert_from_prob_to_state(vprob)
        return vstate

    def get_hidden_prob_given_visible(self, v):
        return tf.sigmoid(2.0 * (tf.matmul(v, self.W) + self.bh))

    def get_visible_prob_given_hidden(self, h):
        return tf.sigmoid(2.0 * (tf.matmul(h, tf.transpose(a=self.W)) + self.bv))

    def convert_from_prob_to_state(self, prob):
        v = prob - tf.random.uniform(tf.shape(input=prob), 0, 1)
        return tf.where(tf.greater_equal(v, tf.zeros_like(v)), tf.ones_like(v), -1 * tf.ones_like(v))

    def get_parameters(self):
        return [self.W, self.bv, self.bh]

    def set_connection(self, connection):
        self.connection = connection

    def make_pickle_object(self, sess):
        temp_rbm = copy.copy(self)
        temp_rbm.W, temp_rbm.bv, temp_rbm.bh = sess.run((self.W, self.bv, self.bh))
        return temp_rbm

    def __str__(self):
        return 'RBM %d-%d' % (self.num_visible, self.num_hidden)

    def to_xml(self):
        stri = ""
        stri += "<machine>\n"
        stri += "\t<type>rbm_real</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<num_visible>%d</num_visible>\n" % self.num_visible
        stri += "\t\t<num_hidden>%d</num_hidden>\n" % self.num_hidden
        stri += "\t\t<density>%d</density>\n" % self.density
        stri += "\t\t<use_bias>%s</use_bias>\n" % str(self.use_bias)
        stri += "\t</params>\n"
        stri += "</machine>\n"
        return stri

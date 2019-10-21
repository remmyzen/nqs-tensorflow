import tensorflow as tf
import copy
import sys
import numpy as np
import os
import pickle
import itertools

class RBMTransfer(object):

    def __init__ (self, machine_target, graph_target, base_model_path, base_model_number=None):
        self.machine_target = machine_target
        self.graph_target = graph_target
        self.base_model_path = base_model_path
        self.base_model_number = base_model_number 
        self.initialize()

    def update_machine(self): 
        self.machine_target.W_array = self.W_transfer
        self.machine_target.bv_array = self.bv_transfer
        self.machine_target.bh_array = self.bh_transfer

    def initialize(self):
        # Get the base model from the path
        self.learner_base = self.get_base_model()
        self.machine_base = self.learner_base.machine
        self.graph_base = self.learner_base.graph

        # Initialize the transferred weight and biases from the target machine 
        if self.learner_base.machine.__class__.__name__ == 'rbm_real_symm':
            self.W_transfer = self.machine_target.W_symm_array 
            self.bv_transfer = self.machine_target.bv_symm_array
            self.bh_transfer = self.machine_target.bh_symm_array

            self.W_base = self.machine_base.W_symm
            self.bv_base = self.machine_base.bv_symm
            self.bh_base = self.machine_base.bh_symm
        
        else:
            self.W_transfer = self.machine_target.W_array 
            self.bv_transfer = self.machine_target.bv_array
            self.bh_transfer = self.machine_target.bh_array

            self.W_base = self.machine_base.W
            self.bv_base = self.machine_base.bv
            self.bh_base = self.machine_base.bh


    def get_base_model(self):
        if self.base_model_number is None:
            dir_names = [int(f) for f in os.listdir(self.base_model_path) if os.path.isdir(self.base_model_path + f)]
            self.base_model_number = max(dir_names)
        self.base_model_path = '%s/%d/model.p' % (self.base_model_path, self.base_model_number)
        base_model = pickle.load(open(self.base_model_path))
        return base_model


    def tiling (self, k_val):
        assert self.machine_target.num_visible >= self.machine_base.num_visible and self.machine_target.num_visible % self.machine_base.num_visible == 0, "Number of visible node in the machine must be larger than or equal to and divisible by the number of visible node in the base machine!"
        assert self.graph_base.length % k_val == 0, "k must be divisible by the number of visible node in base machine!"
        
        p_val = self.graph_target.length / self.graph_base.length

        base_coor = []
        for point in range(self.graph_base.num_points):
            ##### Map old coordinate to the new coordinate which is the old_coor * the k_size
            ## For instance:
            ## 1D from 4 to 8 particles
            ## o-o-o-o  to o-o-o-o-o-o-o-o 
            ## 0 1 2 3  to 0 1 2 3 4 5 6 7
            ## 0 will be transferred to 0
            ## 1 will be transferred to 2 and so on
            ##
            ## 2D from 2x2 to 4x4
            ## 0,0    0,1        0,0     0,1    0,2    0,3
            ##  o------o          o-------o------o------o
            ##  |      |          |      1|1    1|2     |
            ##  o------o     1,0  o-------o------o------o 1,3
            ## 1,0    1,1         |      2|1    2|2     |
            ##               2,0  o-------o------o------o 2,3
            ##                    |      3|1    3|2     |
            ##               3,0  o-------o------o------o 3,3
            ## 0,0 will be transfered to 0,0
            ## 1,0 will be tranferred to 2,0
            ## and so on. 
            ## Similar for 3D
            old_coor = np.array(self.graph_base._point_to_coordinate(point))
                
            ## map the first position of the old coordinate in the base network to the new coordinate in the target network
            new_coor = (old_coor / k_val) * (k_val * p_val) + (old_coor % k_val)
             
            
            ##### Generate all possible combinations for the product 
            ## We want to transfer 0 to 0 and 1 for 1D
            ## and 0,0 to 0,0; 0,1; 1,0; 1,1 for 2D 
            ## We generate all possible combinations for the product
            ## For instance: 
            ## 1D from 4 to 8 particles
            ## old_coor 0 -> new_coor 0 -> to_iter = [[0,1]]
            ## old_coor 2 -> new coor 4 -> to_iter = [[4,5]]
            ## 1D from 4 to 16 particles
            ## old_coor 0 -> new_coor 0 -> to_iter = [[0, 1, 2, 4]
            ## old_coor 2 -> new_coor 8 -> to_iter = [[8, 9, 10, 11]]
            ## 2D from 2x2 to 4x4 particles
            ## old_coor 0,0 -> new_coor 0,0 -> to_iter = [[0,1],[0,1]]
            ## old_coor 1,0 -> new_coor 2,0 -> to_iter = [[2,3],[0,1]]
            ## 3D from 2x2x2 to 4x4x4
            ## old_coor (0,0,0) -> new_coor 0,0,0 -> to_iter=[[0,1], [0,1], [0,1]]
            ## old_coor (1,0,1) -> new_coor 2,0,2 -> to_iter=[[2,3], [0,1], [2,3]]
            ##
            ## because later we will do a product multiply on the to_iter to generate all possible combinations except for 1D
            ## 2D from 2x2 to 4x4 
            ## new_coor 0,0 -> to_iter = [[0,1],[0,1]] do a product multiply
            ## [0,1] x [0,1] = [[0,0], [0,1], [1,0], [1,1]
            ## new_coor 2,0 -> to_iter = [[2,3],[0,1]] do a product multiply
            ## [2,3] x [0,1] = [[2,0], [2,1], [3,0], [3,1]] 
            ## so we get the mapping for transfer
            ## 3D from 2x2x2 to 4x4x4
            ## [2,3] x [0,1] x [2,3] = [2,0,2], [2,0,3], [2,1,2], [2,1,3], ....
            
            to_iter = []
            for dd in range(self.graph_target.dimension):
                temp = []
                for pp in range(p_val):
                    temp.append(new_coor[dd] + pp * k_val)
                to_iter.append(temp)
                
            ### List all combinations to be replaced which is the product that has been explained before
            ## For example in 3d from 2 to 4
            ## old_coor (0,0,0), new coordinates = (0,0,0), (0,0,1), (0,1,0), (0,1,1) ....
            ## old_coor (1,1,1), new_coordinates = (2,2,2), (2,2,3), (2,3,2), .... 
            
            new_coordinates = []
            if self.graph_target.dimension == 1:
                new_coordinates = [[a] for a in to_iter[0]]
            else:
                for kk in to_iter:
                    if len(new_coordinates) == 0:
                        new_coordinates = kk 
                    else: 
                        new_coordinates = [list(cc[0] + [cc[1]]) if isinstance(cc[0], list) else list(cc)  for cc in list(itertools.product(new_coordinates, kk))]


            ## Replace all in the new coordinates with the base coordinates
            ## Connect to new hidden
            for coord in new_coordinates:
                quadrant = [c / self.graph_base.length for c in coord]
                hid_pos = 0
                for ddd in range(self.graph_base.dimension):
                    hid_pos += quadrant[ddd] * (p_val  ** ddd)             

                target_point = self.graph_target._coordinate_to_point(coord)
                
                self.W_transfer[target_point, hid_pos * self.W_base.shape[1] :(hid_pos + 1) * self.W_base.shape[1]] = self.W_base[point, :]
           

        self.update_machine()
        

    def cutpaste(self):
        assert self.machine_target.num_visible >= self.machine_base.num_visible, "Number of visible node in the machine must be larger than or equal to the numbero f visible node in the base machine!"

        for ii in range(self.graph_base.num_points):
            new_coor = self.graph_target._coordinate_to_point(self.graph_base._point_to_coordinate(ii))
            self.W_transfer[new_coor,:self.W_base.shape[1]] = self.W_base[ii]

        self.bv_transfer[:self.bv_base.shape[1]] = self.bv_base
        self.bh_transfer[:self.bh_base.shape[1]]  = self.bh_base
         
        self.update_machine()



        
    

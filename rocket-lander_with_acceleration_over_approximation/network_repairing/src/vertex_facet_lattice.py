import sys
import copy as cp
import numpy as np
import collections as cln

class VFL:
    def __init__(self, lattice, vertices, dim, M, b):
        self.lattice = lattice
        self.vertices = vertices
        self.dim = dim
        self.M = M
        self.b = b
        self.rlvertices = None
        self.base_vertices = None
        self.base_vectors = None


    def compute_real_vertices(self):
        # self.rlvertices = np.dot(self.vertices, self.M.T)+self.b.T
        # self.rlvertices = self.base_vertices +- self.base_vectors
        self.base_vertices = np.dot(self.M, self.vertices.T) + self.b
        self.base_vectors = np.zeros((self.base_vertices.shape[0], 1))


    def to_cuda(self):
        self.is_cuda = True
        self.vertices = self.vertices.cuda()
        self.vertices_init = self.vertices_init.cuda()


    # linear Transformation
    def linearTrans(self, M, b):
        if M.shape[1] != self.M.shape[0]:
            print("dimension is inconsistant")
            sys.exit(1)

        self.M = np.dot(M, self.M)
        self.b = np.dot(M, self.b) + b

        # # update dim
        # if M.shape[0] < self.dim:
        #     self.dim = M.shape[0]
        # set some dim to zero for Relu function

    def map_negative_poly(self, n):
        if self.dim == 0:
            return self

        self.M[n, :] = 0
        self.b[n, :] = 0
        return self

    def single_split_relu(self, idx):
        elements = np.matmul(self.vertices, self.M[idx,:].T)+self.b[idx,:].T
        if np.any(elements==0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements>0)
        positive_id = np.asarray(positive_bool.nonzero()).T
        negative_bool = np.invert(positive_bool)
        negative_id = np.asarray(negative_bool.nonzero()).T

        if len(positive_id)>=len(negative_id):
            less_bool = negative_bool
            more_bool = positive_bool
            flg = 1
        else:
            less_bool = positive_bool
            more_bool = negative_bool
            flg = -1

        vs_facets0 = self.lattice[less_bool]
        vs_facets1 = self.lattice[more_bool]
        vertices0 = self.vertices[less_bool]
        vertices1 = self.vertices[more_bool]
        elements0 = elements[less_bool]
        elements1 = elements[more_bool]

        # t0 = time.time()
        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))

        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])
        if len(new_vs)>50000:
            print('len(new_vs)>50000')

        # self.time0 = time.time() - t0
        # t1 = time.time()
        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:,np.any(vs_facets0,0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = VFL(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == 1:
            subset0.map_negative_poly(idx)

        new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):,0] = True # add hyperplane
        sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        new_vertices1 = np.concatenate((vertices1, new_vs))
        subset1 = VFL(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))
        if flg == -1:
            subset1.map_negative_poly(idx)

        # self.time1 = time.time()-t1
        # subset0.time0 = self.time0
        # subset0.time1 = self.time1
        
        return subset0, subset1


    def single_split(self, A, d):
        A_new = np.dot(A,self.M)
        d_new = np.dot(A, self.b) +d
        elements = np.dot(A_new, self.vertices.T) + d_new
        elements = elements[0]
        if np.all(elements >= 0):
            return None
        if np.all(elements <= 0):
            return self
        if np.any(elements == 0.0):
            sys.exit('Hyperplane intersect with vertices!')

        positive_bool = (elements > 0)
        negative_bool = np.invert(positive_bool)

        vs_facets0 = self.lattice[negative_bool]
        vs_facets1 = self.lattice[positive_bool]
        vertices0 = self.vertices[negative_bool]
        vertices1 = self.vertices[positive_bool]
        elements0 = elements[negative_bool]
        elements1 = elements[positive_bool]

        # t0 = time.time()
        edges = np.dot(vs_facets0.astype(np.float32), vs_facets1.T.astype(np.float32))
        edges_indx = np.array(np.nonzero(edges == self.dim - 1))
        if len(edges_indx[0])+len(edges_indx[1]) == 0:
            sys.exit('Intersected edges are empty!')
        indx0, indx1 = edges_indx[0], edges_indx[1]
        p0s, p1s = vertices0[indx0], vertices1[indx1]
        # ratio0 = (np.dot(A_new, p0s.T) + d_new)[0]
        # ratio1 = (np.dot(A_new, p1s.T) + d_new)[0]
        # alpha = abs(ratio0) / (abs(ratio0) + abs(ratio1))
        elem0, elem1s = elements0[indx0], elements1[indx1]
        alpha = abs(elem0) / (abs(elem0) + abs(elem1s))
        new_vs = p0s + ((p1s - p0s).T * alpha).T
        new_vs_facets = np.logical_and(vs_facets0[indx0], vs_facets1[indx1])

        if len(new_vs)>50000:
            print('len(new_vs)>50000')

        new_vs_facets0 = np.concatenate((vs_facets0, new_vs_facets))
        sub_vs_facets0 = new_vs_facets0[:, np.any(vs_facets0, 0)]
        vs_facets_hp = np.zeros((len(sub_vs_facets0), 1), dtype=bool)
        vs_facets_hp[-len(new_vs):, 0] = True  # add hyperplane
        sub_vs_facets0 = np.concatenate((sub_vs_facets0, vs_facets_hp), axis=1)
        new_vertices0 = np.concatenate((vertices0, new_vs))
        subset0 = VFL(sub_vs_facets0, new_vertices0, self.dim, cp.copy(self.M), cp.copy(self.b))

        # new_vs_facets1 = np.concatenate((vs_facets1, new_vs_facets))
        # sub_vs_facets1 = new_vs_facets1[:, np.any(vs_facets1, 0)]
        # vs_facets_hp = np.zeros((len(sub_vs_facets1), 1), dtype=bool)
        # vs_facets_hp[-len(new_vs):, 0] = True  # add hyperplane
        # sub_vs_facets1 = np.concatenate((sub_vs_facets1, vs_facets_hp), axis=1)
        # new_vertices1 = np.concatenate((vertices1, new_vs))
        # subset1 = VFL(sub_vs_facets1, new_vertices1, self.dim, cp.copy(self.M), cp.copy(self.b))
        #
        # return subset0, subset1
        # return self
        return subset0






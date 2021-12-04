import random, copy, struct
import warnings
import numpy as np
from bitstring import BitArray
import pandas as pd
from collections import *
import pickle
import multiprocessing
from multiprocessing.dummy import Pool
import random
from ex1_functions import *


class MIN_HASH(object):

    def __init__(self, num_perm=128, seed=1, hashfunc=fibonacci_hash_float, vec = [], label = None):

        self._mersenne61 = np.int64((1 << 61) - 1)
        self._max_hash = np.int64((1 << 32) - 1)
        self.seed = seed
        self.num_perm = num_perm
        self.hashfunc = hashfunc
        self.hashvalues = self._init_hashvalues(num_perm)
        self.permutations = self._permutations(num_perm)
        self.label = label
        
        if(len(vec)!=0):
            self.gen_MinHash(vec, label)
            


    def _init_hashvalues(self, num_perm):
        return np.ones(num_perm, dtype=np.uint64)*self._max_hash

    def _permutations(self, num_perm):
        gen = np.random.RandomState(self.seed)
        a_b = []
        for _ in range(num_perm):
            a_b.append((gen.randint(1, self._mersenne61, dtype="int64"), gen.randint(0, self._mersenne61, dtype="int64")))
            
        return np.array(a_b, dtype="int64").T


    def gen_MinHash(self, vec, label):
        self.label = label
        hashed_values = np.array([self.hashfunc(el) for el in vec], dtype="int64")
        a, b = self.permutations
        min_hashed_values = []
        for i, j in zip(a,b):
            min_hashed_values.append((((hashed_values * i)+j)%self._mersenne61))

        phv = np.bitwise_and(np.array(min_hashed_values), self._max_hash).T

        self.hashvalues = phv.min(axis=0)

    def jaccard(self, other):
        return float(np.count_nonzero(self.hashvalues==other.hashvalues)) / float(len(self))
    
    def __len__(self):
    
        return len(self.hashvalues)
    
    def __repr__(self):
        return f'\n======= Label =======\n{self.label.name}\n======= Hash Values =======\n{self.hashvalues}'
    
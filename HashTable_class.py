from bitstring import BitArray
import pandas as pd
from collections import *
import pickle
import multiprocessing
from multiprocessing.dummy import Pool
import random
from ex1_functions import *

class HashTable:
    def __init__(self):
        self.__hash_table = defaultdict(list)
        
    def _generate_hash(self, inp_vector):
        hashVal = 0
        for i in inp_vector:
            hashVal ^= fibonacci_hash_float(i) ^ hashVal
        return hashVal
            
    def setitem(self, vec, label):
        val = self._generate_hash(vec)
        self.hash_table[val].append(label)
        
        
    def getitem(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])
    
    @property
    def hash_table(self):
        return self.__hash_table
    
    @hash_table.setter
    def hash_table(self, value):
        self.__hash_table
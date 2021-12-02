from bitstring import BitArray
import pandas as pd
from collections import *
import pickle
import multiprocessing
from multiprocessing.dummy import Pool
import random
from MIN_HASH_class import MIN_HASH
from HashTable_class import HashTable
from ex1_functions import *

class LSH:
    def __init__(self, minhash_len, num_band):
        
        assert minhash_len % num_band == 0, "the choosen number of band does not hold the following assertion: minhash_len % num_band == 0"
        self.minhash_len = minhash_len
        self.num_band = num_band
        self.hash_tables = list()
        self.minhash_dict = dict()
        for i in range(self.num_band):
            self.hash_tables.append(HashTable())
            
            
    def addMinHash(self, minhash):
        self.minhash_dict[minhash.label] = minhash
        self._create_store_band(minhash.hashvalues, minhash.label)
            
    def _create_store_band(self, vec, label):
        
        row_per_band = self.minhash_len // self.num_band
        
        subVec = []
        
        for i in range(0,self.minhash_len, row_per_band):
            subVec.append(vec[i:i+row_per_band])
        

        for band, table in zip(subVec, self.hash_tables):
            table.setitem(band, label)
            
    
    def _create_band(self, vec):
        
        row_per_band = self.minhash_len // self.num_band
        
        subVec = []
        
        for i in range(0,self.minhash_len, row_per_band):
            subVec.append(vec[i:i+row_per_band])
        
        return subVec
    
            
    def query(self, minhash_query):
        subVec_query = self._create_band(minhash_query.hashvalues)
        match = set()
        similarities = []
        for table, band in zip(self.hash_tables, subVec_query):
            
            key = 0
            for i in band:
                key ^= fibonacci_hash_float(i) ^ key
                
            if(table.hash_table.get(key, "NA") != "NA"):
                match.update(tuple(table.hash_table.get(key)))
                
        for m in match:
            
            similarities.append((self.minhash_dict[m].jaccard(minhash_query), m.name))
            
            similarities = [ i for i in sorted(similarities, reverse=True)]
                
        return similarities
        

        


    def info(self):
        print("Numero di tabelle: " + str(self.num_tables))
        print("Elementi per tabella: " + str(len(self.hash_tables[0].hash_table)))
        
#=================================================================================================  

def query(bands, minhashes, query_minhashes, num_perm = 128):
    
    for b in bands:
        out = []
        lsh = LSH(minhash_len=num_perm, num_band=b)
        print("==========================\nQuery with " + str(b) + " number of bands:\n")
        for i in minhashes:
            lsh.addMinHash(i)

        for query in query_minhashes:
            dfout = pd.DataFrame()
            dfout[query.label.name] = [name[1] for name in lsh.query(query) if len(name) > 0]
            dfout["similarity"] = [sim[0] for sim in lsh.query(query) if len(sim) > 0]
            #out.append((query.label.name, lsh.query(query)))
            dfout.set_index("similarity", inplace=True)
            if(len(dfout) > 0):
                display(dfout)
                print("\n\n")
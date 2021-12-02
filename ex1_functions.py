from bitstring import BitArray
import pandas as pd
from collections import *
import pickle
import multiprocessing
from multiprocessing.dummy import Pool
import random
import numpy as np
from tqdm.notebook import tqdm
from audioSignal_functions import *

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def read_object(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data


def timeOfPeaks(peaks, times):
    timesPeaks = []
    
    for i in peaks:
        timesPeaks.append(times[i])
    
    return timesPeaks


def fibonacci_hash_float(value:float, rand = False, hash_size = 18):

    value = BitArray(float=value, length=64)
    phi = (1 + 5 ** 0.5) / 2
    g = int(2 ** 64 /phi)


    value ^= value >> 61

   
    value = int(g * value.float)

    return int(str(value)[0:hash_size])


def make_fingerprints(audio, duration, hop = 0):
    track, sr, onset_env, peaks = load_audio_picks(audio, duration, HOP_SIZE)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=HOP_SIZE)
    timesPeaks = timeOfPeaks(peaks, times)
    freqsP = [onset_env[i] for i in peaks]
    fingerprints = []
    
    if(hop != 0):
        sec = hop
        time=0
        count=0
        while(sec <= duration):

            idx = 0
            hashVal = 0
            count += 1

            while(time <=sec):

                if(timesPeaks[idx] < sec and timesPeaks[idx] > time):
                    hashVal ^= fibonacci_hash_float(freqsP[idx]) ^ hashVal


                time = timesPeaks[idx]

                if(idx+1 < len(freqsP)):
                    idx += 1
                else:
                    break

            fingerprints.append(hashVal)

            sec += hop

        fingerprints = list(filter(lambda a: a != 0, fingerprints))
    
    else:
        for fr in freqsP:
            fingerprints.append(fr)
   
    return (fingerprints, audio)


def make_all_fingerprints(duration, songs, hop = 0): 
    tempList = list()
    with Pool(multiprocessing.cpu_count()) as pool:
       
         with tqdm(total = len(songs)) as pbar:
            for i, el in enumerate(pool.imap(lambda song: make_fingerprints(song, duration, hop), songs)):
                tempList.append(el)
                pbar.update()
                
    return tempList



def getPeaks(audio, duration):
    _, _, _, peaks = load_audio_picks(audio, duration, HOP_SIZE);
    
    return (peaks, audio)


def get_all_peaks(songs, duration):
    
    peaks_of_songs = []
    
    with Pool(multiprocessing.cpu_count()) as pool:

        with tqdm(total = len(songs)) as pbar:
            for i, el in enumerate(pool.imap(lambda song: getPeaks(song, duration), songs)):
                peaks_of_songs.append(el)
                pbar.update()
                
    return peaks_of_songs


def threshold(b, num_perm = 128):
    
    r = num_perm / b
    t = 1 - ((1/b)**(1/r))
    
        
    return t


def find_band_from_threshold(th, num_perm = 128):
    bands = np.array(range(1,num_perm+1))
    all_thresholds = threshold(bands)
    
    err = []
    for b, t in zip(bands,all_thresholds):
        if(128 % b == 0):
            err.append((abs(t-th), b))
    
    opt_band = min(err)[1]
    
    final_threshold = all_thresholds[opt_band-1]
    
    return opt_band, bands, all_thresholds, final_threshold
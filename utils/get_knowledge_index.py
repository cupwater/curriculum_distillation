import os,sys
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_index(file_in, thr=0.9):
    score = np.loadtxt(file_in)
    max_score = np.max(score, axis=1)
    index = np.where((max_score < thr))
    print('index len: {}'.format(len(index[0])))
    
    out = os.path.join(os.path.dirname(file_in), 'index_lt_{}.txt'.format(thr))
    print('write file to: {}'.format(out))
    np.savetxt(out, index[0])
    


##
if len(sys.argv) == 1 :
    print("Usage: [train_samples_confidence] [threshold]")
    exit(-1)

file1 = sys.argv[1]
thr = sys.argv[2] if len(sys.argv) > 2 else 0.9
save_index(file1, thr)

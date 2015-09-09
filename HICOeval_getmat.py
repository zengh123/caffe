import numpy as np
caffe_root = ''  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
#%matplotlib inline
import os
import scipy.io as sio
os.environ['LMDB_FORCE_CFFI'] = '1'
caffe.set_device(0)
caffe.set_mode_gpu()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print 'prepare net'
net = caffe.Net('models/hico/hico_test.prototxt',
                'models/hico/hico_iter_60000.caffemodel',
                caffe.TEST)
# When using this script, changing the test part of model.prototxt to batch 1
print 'start'
num = 9155 
#num = 36631
all_score = np.zeros((num,600),dtype=np.float32)
ori_label = np.zeros((num,600))
i = 0
for i in xrange(0,num):
    out = net.forward();
    res = net.blobs['fc8'].data
    score = sigmoid(res[0])
    all_score[i,:] = score
    print np.where(score>0.5)
    #i = i+1
    print i      
sio.savemat('test-result.mat', {'all_score':all_score})


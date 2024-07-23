# -*- coding: utf-8 -*-
import pickle as pkl
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os
import numpy.linalg as la
from input_data import preprocess_data,load_sz_data,load_los_data
from tgcn import tgcnCell
#from gru import GRUCell 

from visualization import plot_result,plot_error
from sklearn.metrics import mean_squared_error,mean_absolute_error
#import matplotlib.pyplot as plt
import time
import argparse

time_start = time.time()
# Reset the default graph
tf.reset_default_graph()

# Define command-line arguments using argparse

parser = argparse.ArgumentParser(description='Your description here')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--training_epoch', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--gru_units', type=int, default=64, help='Hidden units of GRU.')
parser.add_argument('--seq_len', type=int, default=12, help='Time length of inputs.')
parser.add_argument('--pre_len', type=int, default=1, help='Time length of prediction.')
parser.add_argument('--train_rate', type=float, default=0.8, help='Rate of training set.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
parser.add_argument('--dataset', type=str, default='sz', help='sz or los.')
parser.add_argument('--model_name', type=str, default='tgcn', help='Model name')
parser.add_argument('--adjacency_matrix', type=str, default='dist', help='Specifies whether to estimate adjacency matrix (gsl, dist, gsldist).')  # gsl, gsldist

args = parser.parse_args()

model_name = args.model_name
data_name = args.dataset
train_rate =  args.train_rate
seq_len = args.seq_len
output_dim = pre_len = args.pre_len
batch_size = args.batch_size
lr = args.learning_rate
training_epoch = args.training_epoch
gru_units = args.gru_units
adj_matrix = args.adjacency_matrix

###### load data ######
data, adj = load_sz_data('sz')
# Drop the first 11 days
data = data.drop(data.index[:1056])

W_est_file_name = f"est_adj/W_est_{data_name}_pre_len{pre_len}.npy"
if 'gsl' in adj_matrix: # gsl or gsldist
    # Load the matrix from the file
    W_est_all = np.load(W_est_file_name)
    if W_est_all.ndim == 2:
        W_est = W_est_all>0
    elif W_est_all.ndim == 3:    
        W_est = np.any(W_est_all>0, axis=2)
    # If gsl method is used, the previous adj is reset to zero
    # else the computed ad learned by gsl is added to adj
    if adj_matrix == 'gsl':
        adj = np.zeros(W_est.shape, dtype=int)

    # Update values in adj based on the condition
    adj[W_est > 0] = 1

    # converting graph to undirected
    # adj = adj + adj.T

    # print('W_est is loaded')

time_len = data.shape[0]
num_nodes = data.shape[1]
data1 =np.mat(data,dtype=np.float32)

#### normalization
max_value = np.max(data1)
data1  = data1/max_value
trainX, trainY, testX, testY = preprocess_data(data1, time_len, train_rate, seq_len, pre_len)

totalbatch = int(trainX.shape[0]/batch_size)
training_data_count = len(trainX)

def TGCN(_X, _weights, _biases):
    ###
    cell_1 = tgcnCell(gru_units, adj, num_nodes=num_nodes)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell_1], state_is_tuple=True)
    _X = tf.unstack(_X, axis=1)
    outputs, states = tf.nn.static_rnn(cell, _X, dtype=tf.float32)
    m = []
    for i in outputs:
        o = tf.reshape(i,shape=[-1,num_nodes,gru_units])
        o = tf.reshape(o,shape=[-1,gru_units])
        m.append(o)
    last_output = m[-1]
    output = tf.matmul(last_output, _weights['out']) + _biases['out']
    # output = tf.reshape(output,shape=[-1,num_nodes,pre_len])
    output = tf.reshape(output,shape=[-1,num_nodes, 1])
    output = tf.transpose(output, perm=[0,2,1])
    output = tf.reshape(output, shape=[-1,num_nodes])
    return output, m, states
        
###### placeholders ######
inputs = tf.placeholder(tf.float32, shape=[None, seq_len, num_nodes])
# labels = tf.placeholder(tf.float32, shape=[None, pre_len, num_nodes]) M. Amintoosi
labels = tf.placeholder(tf.float32, shape=[None, 1, num_nodes])

# Graph weights
# weights = {
#     'out': tf.Variable(tf.random_normal([gru_units, pre_len], mean=1.0), name='weight_o')}
# biases = {
#     'out': tf.Variable(tf.random_normal([pre_len]),name='bias_o')}
weights = {
    'out': tf.Variable(tf.random_normal([gru_units, 1], mean=1.0), name='weight_o')}
biases = {
    'out': tf.Variable(tf.random_normal([1]),name='bias_o')}


if model_name == 'tgcn':
    pred,ttts,ttto = TGCN(inputs, weights, biases)

y_pred = pred
      

###### optimizer ######
lambda_loss = 0.0015
Lreg = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
label = tf.reshape(labels, [-1,num_nodes])
##loss
loss = tf.reduce_mean(tf.nn.l2_loss(y_pred-label) + Lreg)
##rmse
error = tf.sqrt(tf.reduce_mean(tf.square(y_pred-label)))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)

###### Initialize session ######
variables = tf.global_variables()
saver = tf.train.Saver(tf.global_variables())  
#sess = tf.Session()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())

out = 'out/%s'%(model_name)
#out = 'out/%s_%s'%(model_name,'perturbation')
path1 = '%s_%s_lr%r_batch%r_unit%r_seq%r_pre%r_epoch%r_adj-%s'%(model_name,data_name,lr,batch_size,gru_units,seq_len,pre_len,training_epoch,adj_matrix)
path = os.path.join(out,path1)
if not os.path.exists(path):
    os.makedirs(path)
    
###### evaluation ######
def evaluation(a,b):
    rmse = math.sqrt(mean_squared_error(a,b))
    mae = mean_absolute_error(a, b)
    F_norm = la.norm(a-b,'fro')/la.norm(a,'fro')
    r2 = 1-((a-b)**2).sum()/((a-a.mean())**2).sum()
    var = 1-(np.var(a-b))/np.var(a)
    return rmse, mae, 1-F_norm, r2, var
 
   
x_axe,batch_loss,batch_rmse,batch_pred = [], [], [], []
test_loss,test_rmse,test_mae,test_acc,test_r2,test_var,test_pred = [],[],[],[],[],[],[]
# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['Iter', 'train_rmse', 'test_loss', 'test_rmse', 'test_acc'])

for epoch in range(training_epoch):
    for m in range(totalbatch):
        mini_batch = trainX[m * batch_size : (m+1) * batch_size]
        mini_label = trainY[m * batch_size : (m+1) * batch_size]
        _, loss1, rmse1, train_output = sess.run([optimizer, loss, error, y_pred],
                                                 feed_dict = {inputs:mini_batch, labels:mini_label})
        batch_loss.append(loss1)
        batch_rmse.append(rmse1 * max_value)

     # Test completely at every epoch
    loss2, rmse2, test_output = sess.run([loss, error, y_pred],
                                         feed_dict = {inputs:testX, labels:testY})
    test_label = np.reshape(testY,[-1,num_nodes])
    rmse, mae, acc, r2_score, var_score = evaluation(test_label, test_output)
    test_label1 = test_label * max_value
    test_output1 = test_output * max_value
    test_loss.append(loss2)
    test_rmse.append(rmse2 * max_value) # M. Amintoosi, rmse->rmse2
    test_mae.append(mae * max_value)
    test_acc.append(acc)
    test_r2.append(r2_score)
    test_var.append(var_score)
    test_pred.append(test_output1)
    
    # df = df.append({'Iter': epoch, 'train_rmse': round(train_rmse, 4), 'test_loss': round(loss2, 4), 'test_rmse': round(rmse, 4), 'test_acc': round(acc, 4)}, ignore_index=True)

    print('Iter:{}'.format(epoch),
          'train_rmse:{:.4}'.format(batch_rmse[-1]),
          'test_loss:{:.4}'.format(loss2),
          'test_rmse:{:.4}'.format(rmse2 * max_value), # M. Amintoosi, rmse->rmse2 * max_value
          'test_acc:{:.4}'.format(acc))
    
    if (epoch % 50 == 0):        
        saver.save(sess, path+'/model_100/TGCN_pre_%r'%epoch, global_step = epoch)
        
time_end = time.time()
print(time_end-time_start,'s')

############## visualization ###############
b = int(len(batch_rmse)/totalbatch)
batch_rmse1 = [i for i in batch_rmse]
train_rmse = [(sum(batch_rmse1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]
batch_loss1 = [i for i in batch_loss]
train_loss = [(sum(batch_loss1[i*totalbatch:(i+1)*totalbatch])/totalbatch) for i in range(b)]

index = test_rmse.index(np.min(test_rmse))
test_result = test_pred[index]
# var = pd.DataFrame(test_result)
# var.to_csv(path+'/test_result.csv',index = False,header = False)

# Create dataframes for test_result and test_label1
df_test_result = pd.DataFrame(test_result)
df_test_label1 = pd.DataFrame(test_label1)

# Specify the path for the csv file
file_name = path+'/test_result.xlsx'

# Write to the csv file with two sheets
with pd.ExcelWriter(file_name) as writer:
    df_test_result.to_excel(writer, sheet_name='pred', index=False, header=False)
    df_test_label1.to_excel(writer, sheet_name='true', index=False, header=False)

plot_result(test_result,test_label1,path)
plot_error(train_rmse,train_loss,test_rmse,test_acc,test_mae,path)

print('min_rmse:%r'%(np.min(test_rmse)),
      'min_mae:%r'%(test_mae[index]),
      'max_acc:%r'%(test_acc[index]),
      'r2:%r'%(test_r2[index]),
      'var:%r'%test_var[index])

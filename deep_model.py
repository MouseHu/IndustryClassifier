import pandas as pd
# Import the relevant libraries
import math
import numpy as np
import os
from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)

import cntk as C
import cntk.tests.test_utils
import pickle
from cntk import sequence
from cntk import load_model

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components


from cntk.device import try_set_default_device, gpu,cpu
try_set_default_device(gpu(0))

# number of words in vocab, slot labels, and intent labels
vocab_size = 80000; num_labels = 19
title_size = 52000
body_size = 210000
# model dimensions
input_dim  = vocab_size
label_dim  = num_labels
emb_dim    = 300
hidden_dim = 200
max_length = 50
model = 'cnn'
embed_file = None
industry=[i.rstrip("\n") for i in open("news/industry.wl").readlines()]

if embed_file:
    with open(embedfile,'rb') as handle:
        embedding=pickle.load(handle)
    
# Create the containers for input feature (x) and the label (y)
i1_axis = C.Axis.new_unique_dynamic_axis('1')
i2_axis = C.Axis.new_unique_dynamic_axis('2')
xb = C.sequence.input(shape=body_size, is_sparse=True, sequence_axis=i1_axis, name='xb')
xt = C.sequence.input(shape=title_size, is_sparse=True, sequence_axis=i2_axis, name='xt')
x = C.sequence.input_variable(vocab_size)

y = C.input_variable(num_labels)

def BiRecurrence(fwd, bwd):
    F = C.layers.Recurrence(fwd)
    G = C.layers.Recurrence(bwd, go_backwards=True)
    x = C.placeholder()
    apply_x = C.splice(sequence.last(F(x)), sequence.first(G(x)),name='h2')
    return apply_x


def create_model():
    return {
        'cnn':create_model_cnn(),
        'gru':create_model_gru(),
        'lstm':create_model_lstm(),
        'cnn_body':create_model_cnn_body(),
        'cnn_2fold':create_model_cnn_2fold()
    }[model]
def create_model_gru(embed=False):
    with C.layers.default_options(initial_state=0.1):
        if embed:
            h1= C.layers.Sequential([
            C.layers.Embedding(emb_dim,name='embed',init=embedding),
            C.layers.BatchNormalization(),
            C.layers.Stabilizer()])
        else:
            h1= C.layers.Sequential([
            C.layers.Embedding(emb_dim,name='embed'),
            C.layers.BatchNormalization(),
            C.layers.Stabilizer()])

        h2=BiRecurrence(C.layers.GRU(hidden_dim),C.layers.GRU(hidden_dim))(h1)
        #h3=C.sequence.last(h2)
        h4=C.layers.Dense(num_labels, name='classify')(h2)
    return h4
def create_model_lstm(embed = False):
    with C.layers.default_options(initial_state=0.1):
        if embed:
            h1= C.layers.Sequential([
            C.layers.Embedding(emb_dim,name='embed',init=embedding),
            C.layers.BatchNormalization(),
            C.layers.Stabilizer()])
        else:
            h1= C.layers.Sequential([
            C.layers.Embedding(emb_dim,name='embed'),
            C.layers.BatchNormalization(),
            C.layers.Stabilizer()])       
        h2=BiRecurrence(C.layers.LSTM(hidden_dim),C.layers.LSTM(hidden_dim))(h1)
        h4=C.layers.Dense(num_labels, name='classify')(h2)
    return h4
def create_model_cnn_2fold():
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    #
    with C.layers.default_options(initial_state=0.1):


        #h1_1= C.layers.Embedding(weights=embedding,name='embed')(x)#
        h1_1= C.layers.Embedding(300,name='embed')(x)
        h1_2= C.layers.Embedding(300,name='embed')(x)#init=embedding,
        h1 = C.splice(h1_1,h1_2,axis = 0)
        #bn = C.layers.BatchNormalization(name='bn')(h1)
        to_static= C.layers.PastValueWindow(window_size=max_length, axis=-2)(h1)[0]

        #value,valid = to_static(h1)

        filter_num=100

        h2_1=C.layers.Convolution((3,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)
        h2_2=C.layers.Convolution((4,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)
        h2_3=C.layers.Convolution((5,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)

        h3_1=C.layers.MaxPooling((max_length-2,1),name='pooling')(h2_1)
        h3_2=C.layers.MaxPooling((max_length-3,1),name='pooling')(h2_2)
        h3_3=C.layers.MaxPooling((max_length-4,1),name='pooling')(h2_3)

        h3=C.splice(h3_2,h3_1,h3_3,axis=0)
        drop1 =C.layers.Dropout(0.5)(h3)
        h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4

def create_model_cnn(embed = False):
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    #
    with C.layers.default_options(initial_state=0.1):

       
        if embed:
            h1= C.layers.Embedding(300,init=embedding,name='embed')(x)#
        else:
            h1= C.layers.Embedding(300,name='embed')(x)#init=embedding,
        
        #bn = C.layers.BatchNormalization(name='bn')(h1)
        to_static= C.layers.PastValueWindow(window_size=max_length, axis=-2)(h1)[0]

        #value,valid = to_static(h1)

        filter_num=100

        h2_1=C.layers.Convolution((3,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)
        h2_2=C.layers.Convolution((4,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)
        h2_3=C.layers.Convolution((5,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(to_static)
    
        h3_1=C.layers.MaxPooling((max_length-2,1),name='pooling')(h2_1)
        h3_2=C.layers.MaxPooling((max_length-3,1),name='pooling')(h2_2)
        h3_3=C.layers.MaxPooling((max_length-4,1),name='pooling')(h2_3)
        #h2=BiRecurrence(C.layers.LSTM(hidden_dim), C.layers.LSTM(hidden_dim))(h1)
        h3=C.splice(h3_2,h3_1,h3_3,axis=0)
        h4=C.layers.Dense(num_labels,name='hidden')(h3)

    return h4

def create_model_cnn_body():

    with C.layers.default_options(initial_state=0.1):

       

        h1t= C.layers.Embedding(300,name='embed')(xb)#init=embedding,
        h1b= C.layers.Embedding(300,name='embed')(xt)#init=embedding,
        bnb = C.layers.BatchNormalization(name='bn')(h1b)
        bnt = C.layers.BatchNormalization(name='bn')(h1t)
        to_static_t= C.layers.PastValueWindow(window_size=max_length, axis=-2)(bnt)[0]
        to_static_b= C.layers.PastValueWindow(window_size=max_length*4, axis=-2)(bnb)[0]


        h2_1t=C.layers.Convolution((1,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_t)
        h2_2t=C.layers.Convolution((2,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_t)
        h2_3t=C.layers.Convolution((3,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_t)
        
        h2_1b=C.layers.Convolution((1,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_b)
        h2_2b=C.layers.Convolution((2,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_b)
        h2_3b=C.layers.Convolution((3,emb_dim),num_filters=100,reduction_rank=0,activation=C.relu)(to_static_b)

        h3_2t=C.layers.MaxPooling((max_length-1,1),name='pooling')(h2_2t)
        h3_1t=C.layers.MaxPooling((max_length,1),name='pooling')(h2_1t)
        h3_3t=C.layers.MaxPooling((max_length-2,1),name='pooling')(h2_3t)
        
        h3_2b=C.layers.MaxPooling((max_length-1,1),name='pooling')(h2_2b)
        h3_1b=C.layers.MaxPooling((max_length,1),name='pooling')(h2_1b)
        h3_3b=C.layers.MaxPooling((max_length-2,1),name='pooling')(h2_3b)

        h3=C.splice(h3_2t,h3_1t,h3_3t,h3_2b,h3_1b,h3_3b,axis=0)
        
        h4=C.layers.Dense(hidden_dim, activation=C.relu,name='hidden')(h3)
        drop2 = C.layers.Dropout(0.5)(h4)

        h5=C.layers.Dense(num_labels,name='classify')(drop2)

    return h5





def create_reader(path, is_training,is_body=False):
    if is_body:
        return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         title         = C.io.StreamDef(field='S0', shape=title_size,  is_sparse=True),
         industry        = C.io.StreamDef(field='S1', shape=num_labels, is_sparse=True),
         body        = C.io.StreamDef(field='S2', shape=body_size, is_sparse=True),
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)
    else:
        return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(
         title         = C.io.StreamDef(field='S0', shape=vocab_size,  is_sparse=True),
         industry        = C.io.StreamDef(field='S1', shape=num_labels, is_sparse=True),
         
     )), randomize=is_training, max_sweeps = C.io.INFINITELY_REPEAT if is_training else 1)



def create_criterion_function(model):
    labels = C.placeholder(name='labels')
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels)
    return C.combine ([ce, errs]) # (features, labels) -> (loss, metric)

def create_criterion_function_preferred(model, labels):
    ce   = C.cross_entropy_with_softmax(model, labels)
    errs = C.classification_error      (model, labels,topN=1)
    return ce, errs # (model, labels) -> (loss, error metric)
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def evaluate(reader,model_func,is_body=False):#cal precision and recall

    if is_body:
        test_xt = C.sequence.input_variable(title_size)
        
    else:
        test_xt = C.sequence.input_variable(vocab_size)
         
    test_xb = C.sequence.input_variable(body_size)
    test_y = C.input_variable(num_labels)
    model=model_func(x)
    # Create the loss and error functions
    loss, label_error = create_criterion_function_preferred(model, y)
    
    # Assign the data fields to be read from the input
    data_map={x: reader.streams.title, y: reader.streams.industry}
    
    confuse=np.zeros((num_labels,num_labels))
    count=0
    while True:
        data = reader.next_minibatch(2048)  # fetch minibatch
        if not data:
            break
            
        for key in data.keys():
            if(key.m_name=="title"):   
                test_xt=data[key]            
            if(key.m_name=="industry"):      
                test_y=data[key]
            if(key.m_name=="body"): 
                test_xb=data[key]
        #print(data)   
        if is_body:
            output=z(x).eval({xt:test_xt,xb:test_xb}).argmax(axis=1)
        else:
            output=z(x).eval({x:test_xt}).argmax(axis=1)
       
        gt=C.squeeze(C.argmax(y)).eval({y:test_y}).astype(int)#.as_sequences(test_y)[0].indices[0]
        confuse+=fast_hist(output,gt,num_labels)
        count+=1

    precision=np.diag(confuse)/np.sum(confuse,axis=0)
    recall = np.diag(confuse)/np.sum(confuse,axis=1)
    accuarcy = np.diag(confuse).sum() / confuse.sum()
    aver_precision=np.nanmean(precision)
    aver_recall = np.nanmean(recall)
   
    print("Precision:{} Recall:{} Acc:{}".format(aver_precision,aver_recall,accuarcy))
    return accuarcy

def do_test(input_file,is_body=False):
    reader = create_reader(input_file,is_training=False)
    acc = my_evaluate(reader, z,is_body)    
 
    return acc

def train(reader, model_func,epoch_size, max_epochs=30,is_body=False,data_tag = "180"):
    global model
    criterion = create_criterion_function(create_model())
    
    if model in ["lstm","gru"]:
        criterion.replace_placeholders({criterion.placeholders[1]: C.input_variable(num_labels)})
    else:
        criterion.replace_placeholders({criterion.placeholders[0]: C.input_variable(num_labels)})

    if is_body:
        model = model_func(xb,xt)
        data_map={xb: reader.streams.body,xt: reader.streams.title, y: reader.streams.industry}
    else:
        model = model_func(x)
        data_map={x: reader.streams.title, y: reader.streams.industry}
    
    loss, label_error = create_criterion_function_preferred(model, y)

    # training config

    minibatch_size = 2048


    lr_schedule = C.learning_parameter_schedule([3e-4*minibatch_size]*5+[1e-4*minibatch_size], epoch_size=epoch_size)

    # Momentum schedule
    momentums = C.momentum_schedule(0.99, minibatch_size=minibatch_size)

    # We use a the Adam optimizer which is known to work well on this dataset
    # Feel free to try other optimizers from
    # https://www.cntk.ai/pythondocs/cntk.learner.html#module-cntk.learner
    learner = C.adam(parameters=model.parameters,#tuple([x for  x in model.parameters][:-1]),
                     lr=lr_schedule,
                     momentum=momentums,
                     gradient_clipping_threshold_per_sample=15,
                     gradient_clipping_with_truncation=True,
                     l2_regularization_weight=0.0001)

    # Setup the progress updater
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)

    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs)

    # Instantiate the trainer
    trainer = C.Trainer(model, (loss, label_error), learner, progress_printer)

    # process minibatches and perform model training
    C.logging.log_number_of_parameters(model)

    # Assign the data fields to be read from the input
    
    
    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = reader.next_minibatch(minibatch_size, input_map= data_map)  # fetch minibatch
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()
        if epoch%2==0:
            acc=do_test(is_body)
            z.save("news/Models/{}_Model_{}_epoch{}_acc{}.dnn".format(m.upper(),data_tag,epoch/2,acc))
            
def do_train(m,train_file,tag,epoch_size):
    global z
    global model
    model = m
    is_body = m.find('body')>=0
    print(m)
    z = create_model()
    #z= load_model("news/Models/LSTM_Model_180_Autosave_epoch3.0_acc0.7861713891009139.dnn")
    #do_test()
    print(z.parameters)                                                                               
    reader = create_reader(train_file, is_training=True,is_body=is_body)
    train(reader,z,epoch_size,is_body=is_body,data_tag=tag)
    z.save("news/Models/{}Model_{}_final.dnn".format(m.upper()))
def init_param_setting(vocabulary= 100000,industry = 19,title=0,body = 0):
    global vocab_size,title_size,body_size,num_labels
    vocab_size,num_labels,title_size,body_size = vocabulary,indsutry,title,body
def super_param_setting(embed=300,hidden=200,sentence=50):
    global emb_dim,hidden_dim,max_length
    emb_dim,hidden_dim,max_length = embed,hidden,sentence
    
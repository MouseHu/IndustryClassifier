from __future__ import print_function # Use a function definition from future version (say 3.x from 2.7 interpreter)
import pandas as pd
import math
import numpy as np
import os
import time 

import cntk as C
import cntk.tests.test_utils
import pickle
import random
from cntk import sequence
from cntk import load_model
from cntk.device import try_set_default_device, gpu,cpu
from scipy.sparse import csr_matrix
from data_processor import *
from cntk.layers import *
from cntk.layers.typing import *
from gensim.models import Word2Vec

cntk.tests.test_utils.set_device_from_pytest_env() # (only needed for our build system)
C.cntk_py.set_fixed_random_seed(1) # fix a random seed for CNTK components
try_set_default_device(gpu(0))

def load_data_body(input_file,title_dict,body_dict,industry_dict,raw=False):
    data = open(input_file, encoding = "utf-8").readlines()
    
    data_title = np.zeros((len(data),max_length_title),dtype = np.float32)
    data_body  = np.zeros((len(data),max_length_body),dtype = np.float32)
    data_label = np.zeros((len(data),1),dtype = np.float32)
    
    for index,line in enumerate(data):
        row = line.strip("\n").split("\t")        
        if not raw:
            title    =  row[0]
            body     =  row[1]     
        else:
            title    =  tokenize(row[0])
            body     =  tokenize(row[1])
            #industry =  tokenize(row[2])
        industry =  row[2]
        
        for jndex,token in enumerate(title.split(" ")):
            if jndex>=max_length_title:
                break
            data_title[index,jndex]=title_dict.get(token,len(title_dict)-1)
            
        for jndex,token in enumerate(body.split(" ")):
            if jndex>=max_length_body:
                break
            data_body[index,jndex]=body_dict.get(token,len(body_dict)-1)
            
        data_label[index] = industry_dict.get(industry,len(industry_dict))
    return data_title,data_body,data_label

def load_data_dynamic(input_file,title_dict,industry_dict,raw=False):
    data = open(input_file, encoding = "utf-8").readlines()
    
    data_title =[ [] for x in range(len(data))]#np.zeros((len(data),max_length_title),dtype = np.float32)
    data_label = np.zeros((len(data),1),dtype = np.float32)
    
    
    for index,line in enumerate(data):
        row = line.strip("\n").split("\t")
        if raw:
            title    =  tokenize(row[0])
        else:
            title    =  row[0]
        industry =  row[1]
        
        for jndex,token in enumerate(title.split(" ")):
            if jndex>=max_length_title:
                break
            data_title[index].append(title_dict.get(token,len(title_dict)-1))
        while len(data_title[index])<5:
            data_title[index].append(len(title_dict)-1)
        data_label[index] = industry_dict.get(industry,len(industry_dict))
    data_title = [ np.array(x) for x in data_title]
    return data_title,data_label

def load_data_static(input_file,title_dict,industry_dict,raw=False):
    data = open(input_file, encoding = "utf-8").readlines()
    
    data_title =np.zeros((len(data),max_length_title),dtype = np.float32)
    data_label = np.zeros((len(data),1),dtype = np.float32)
    
    
    for index,line in enumerate(data):
        row = line.strip("\n").split("\t")
        if raw:
            title    =  tokenize(row[0])
        else:
            title    =  row[0]
        industry =  row[1]
        
        for jndex,token in enumerate(title.split(" ")):
            if jndex>=max_length_title:
                break
            data_title[index,jndex]=title_dict.get(token,len(title_dict)-1)
        data_label[index] = industry_dict.get(industry,len(industry_dict))
    
    return data_title.tolist(),data_label.tolist()

def load_embedding(dict_file,embedding_model_file):
    model = Word2Vec.load(embedding_model_file)
    dict_list = [x.strip("\n") for x in open(dict_file,encoding = 'utf-8').readlines()]
    embedding = np.zeros((len(dict_list),emb_dim))
    count = 0
    for i,w in enumerate(dict_list):
        try:
            vec = model.wv[w]
        except:
            vec=model.wv["UNK"]
            count+=1
        embedding[i] =vec
    print(count)
    return embedding


def batch_iter(data,batch_size, num_epochs, shuffle=True):
    # Generates a batch iterator for a dataset.
    data_size = len(data)
    num_batches_per_epoch = int((data_size-1)/batch_size) + 1
    print('data_size: ', data_size, 'batch_size: ', batch_size, 'num_batches_per_epoch: ', num_batches_per_epoch)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield data[start_index:end_index]
            

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def test(batch_size,model,data):
    print("Testing...")
    scores = model(input_xt)
    predict = C.argmax(scores,axis = 0)
    confuse = np.zeros((num_labels,num_labels))

    test_data_title,test_data_label = data
    batches = batch_iter(list(zip(test_data_title,test_data_label)), batch_size, 1)
    
    for batch in batches:
        batch_data_title,batch_data_label = zip(*batch) 
        
        batch_data_title = list(batch_data_title)
        #print(type(batch_data_title))
        output = np.array(predict.eval({input_xt: batch_data_title}),dtype=np.int)
        gt = np.array(batch_data_label,dtype=np.int)
        confuse+=fast_hist(output,gt,num_labels)
        
    precision=np.diag(confuse)/np.sum(confuse,axis=0)
    recall = np.diag(confuse)/np.sum(confuse,axis=1)
    accuracy = np.diag(confuse).sum() / confuse.sum()
    aver_precision=np.nanmean(precision)
    aver_recall = np.nanmean(recall)
   
    print("Precision:{} Recall:{} Acc:{}".format(aver_precision,aver_recall,accuracy))
    return accuracy

def test_body(batch_size,model,data):
    print("Testing...")
    scores = model(input_xt,input_xb)
    predict = C.argmax(scores,axis = 0)
    confuse = np.zeros((num_labels,num_labels))
    #C.element_add(input_y,C.element_times(predict,C.Constant([nums_labels])))
    test_data_title,test_data_body,test_data_label = data
    batches = batch_iter(list(zip(test_data_title,test_data_body,test_data_label)), batch_size, 1)
    
    for batch in batches:
        batch_data_title,batch_data_body,batch_data_label = zip(*batch)
        
        batch_data_body = list(batch_data_body)
        batch_data_title = list(batch_data_title)
        output = np.array(predict.eval({input_xb: batch_data_body,input_xt: batch_data_title}),dtype=np.int)
        gt = np.array(batch_data_label,dtype=np.int)
        confuse+=fast_hist(output,gt,num_labels)
    precision=np.diag(confuse)/np.sum(confuse,axis=0)
    recall = np.diag(confuse)/np.sum(confuse,axis=1)
    accuracy = np.diag(confuse).sum() / confuse.sum()
    aver_precision=np.nanmean(precision)
    aver_recall = np.nanmean(recall)
   
    print("Precision:{} Recall:{} Acc:{}".format(aver_precision,aver_recall,accuracy))
    return accuracy

def train(model,train_data,num_epochs,learning_rate,batch_size,tag="CNN",l2_weight=0,show_count =1000,do_test=True):
    
    #print(C.logging.get_node_outputs(model))
    scores = model(input_xt)

    loss =C.reduce_mean(C.losses.cross_entropy_with_softmax(scores, input_y_one_hot))
    
    # Training
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    #learner = C.adam(scores.parameters, lr=lr_schedule, momentum=0.9,l2_regularization_weight=0)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
    momentums = C.momentum_schedule(0.99, minibatch_size=batch_size)
    learner = C.adam(parameters=scores.parameters,#model.parameters,
                     lr=lr_schedule,
                     momentum=momentums,
                     gradient_clipping_threshold_per_sample=15,
                     gradient_clipping_with_truncation=True,
                     l2_regularization_weight=l2_weight)
    trainer = C.Trainer(scores, (loss), [learner], progress_printer)
    
    train_data_title,train_data_label = train_data
    batches = batch_iter(list(zip(train_data_title,train_data_label)), batch_size, num_epochs)

    # training loop
    count = 0
    t = time.time()
    for batch in batches:
        count += 1
        batch_data_title,batch_data_label = zip(*batch)
        batch_data_title = list(batch_data_title)
        batch_data_label = list(batch_data_label)
        #print(type(batch_data_title),type(batch_data_title[0]),batch_data_title[0])
        trainer.train_minibatch({input_xt: batch_data_title, input_y: batch_data_label})
        if count%show_count== 0:
            print("batch count:{} elapse time:{}".format(count,time.time()-t))
            t=time.time()
            if do_test:
                acc1=test(batch_size,model,test_data)
                model.save('./model/{}_acc{:.3f}.dnn'.format(tag,acc1))
            else:
                model.save('./model/{}.dnn'.format(tag))
            
def train_body(model,train_data,num_epochs,learning_rate,batch_size,l2_weight=0,tag = "cnn",show_count =1000,do_test=True):
    
    #learning_rate *= batch_size
    #model = model_func()
    #print(C.logging.get_node_outputs(model))
    scores = model(input_xt,input_xb)

    loss =C.reduce_mean(C.losses.cross_entropy_with_softmax(scores, input_y_one_hot))
    
    # Training
    lr_schedule = C.learning_parameter_schedule(learning_rate)
    #learner = C.adam(scores.parameters, lr=lr_schedule, momentum=0.9,l2_regularization_weight=0)
    progress_printer = C.logging.ProgressPrinter(tag='Training', num_epochs=num_epochs)
    momentums = C.momentum_schedule(0.99, minibatch_size=batch_size)
    learner = C.adam(parameters=scores.parameters,#model.parameters,
                     lr=lr_schedule,
                     momentum=momentums,
                     gradient_clipping_threshold_per_sample=15,
                     gradient_clipping_with_truncation=True,
                     l2_regularization_weight=l2_weight)
    trainer = C.Trainer(scores, (loss), [learner], progress_printer)
    
    train_data_title,train_data_body,train_data_label = train_data
    batches = batch_iter(list(zip(train_data_title,train_data_body,train_data_label)), batch_size, num_epochs)

    # training loop
    count = 0
    t = time.time()
    for batch in batches:
        count += 1
        batch_data_title,batch_data_body,batch_data_label = zip(*batch)
        batch_data_body = list(batch_data_body)
        batch_data_title = list(batch_data_title)
        batch_data_label = list(batch_data_label)
        
        trainer.train_minibatch({input_xb: batch_data_body,input_xt: batch_data_title, input_y: batch_data_label})
        if count%show_count== 0:
            print("batch count:{} elapse time:{}".format(count,time.time()-t))
            t=time.time()
            if do_test:
                acc1=test_body(batch_size,model,test_data)
                model.save('./model/{}_acc{:.3f}.dnn'.format(tag,acc1))
            else:
                model.save('./model/{}.dnn'.format(tag))
            
def inference(model,val_doc_file,output_file,title_dict,industry_file,dynamic=False):
    
    scores = model(input_xt)
    predict = C.argmax(scores,axis = 0)
    probability = C.reduce_max(C.softmax(scores),axis = 0)
    
    industry = [x.strip("\n") for x in open(industry_file,encoding ="utf-8").readlines()]
    val_doc = open(val_doc_file,encoding = "utf-8")
    output = open(output_file,"w",encoding = "utf-8")
    val_doc = [tokenize(x.strip("\n").split("\t")[0]) for x in val_doc.readlines()]
    #print(val_doc[0:5])
    if dynamic:
        data_title = [[] for x  in range(len(val_doc))]
    
        for index,title in enumerate(val_doc):           
            for jndex,token in enumerate(title.split(" ")):
                if jndex>=max_length_title:
                    break
                data_title[index].append(title_dict.get(token,len(title_dict)-1))
            while len(data_title[index])<5:
                data_title[index].append(len(title_dict)-1) 
    else:
        data_title =np.zeros((len(val_doc),max_length_title),dtype = np.float32)
        for index,title in enumerate(val_doc):
            #title    =  row[0]     
            for jndex,token in enumerate(title.split(" ")):
                if jndex>=max_length_title:
                    break
                data_title[index,jndex]=title_dict.get(token,len(title_dict)-1)
    batches = batch_iter(data_title, batch_size, 1,shuffle =False)
    for batch in batches:
        batch_data_title = batch
        pred = np.array(predict.eval({input_xt: batch_data_title}),dtype=np.int)
        prob = np.array(probability.eval({input_xt: batch_data_title}),dtype=np.float32)
        #gt = np.array(batch_data_label,dtype=np.int)
        #confuse+=fast_hist(output,gt,num_labels)
        for pre,pro in list(zip(pred,prob)):
            output.write("\t".join([str(industry[int(pre)]),str(pro[0])])+"\n")
    output.close()

def inference_body(model,val_doc_file,output_file,title_dict,body_dict,industry_file):
    
    scores = model(input_xt,input_xb)
    predict = C.argmax(scores,axis = 0)
    probability = C.reduce_max(C.softmax(scores),axis = 0)
    
    industry = [x.strip("\n") for x in open(industry_file,encoding ="utf-8").readlines()]
    
    val_doc = open(val_doc_file,encoding = "utf-8").readlines()
    output = open(output_file,"w",encoding = "utf-8")
    
    val_title = [tokenize(x.strip("\n").split("\t")[0]) for x in val_doc]
    val_body = [tokenize(x.strip("\n").split("\t")[1]) for x in val_doc]
    
    data_title = np.zeros((len(val_title),max_length_title),dtype = np.float32)
    data_body= np.zeros((len(val_body),max_length_body),dtype = np.float32)
    
    for index,title in enumerate(val_title):       
        for jndex,token in enumerate(title.split(" ")):
            if jndex>=max_length_title:
                break
            data_title[index,jndex]=title_dict.get(token,len(title_dict)-1)
    for index,body in enumerate(val_body):       
        for jndex,token in enumerate(body.split(" ")):
            if jndex>=max_length_body:
                break
            data_body[index,jndex]=body_dict.get(token,len(body_dict)-1)

    batches = batch_iter(list(zip(data_title,data_body)), batch_size, 1,shuffle=False)
    for batch in batches:

        batch_data_title,batch_data_body = zip(*batch)
        pred = np.array(predict.eval({input_xt: np.array(batch_data_title),input_xb: np.array(batch_data_body)}))
        prob = np.array(probability.eval({input_xt: np.array(batch_data_title),input_xb: np.array(batch_data_body)}),dtype=np.float32)

        for pre,pro in list(zip(pred,prob)):
            
            output.write("\t".join([str(industry[int(pre)]),str(pro[0])])+"\n")
    output.close()
    print("predict finished.")
    
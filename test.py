from cntk import load_model

import sys,getopt
from deepmodel_util import *
from deepmodel_definition import *
import deepmodel_util
import deepmodel_definition

input_test_file = ""
tag = "cnn"
prefix = ""
num_labels = 19
#input_lr = 5e-4
input_batch_size = 100
do_test=False
suffix = ""

opts,args = getopt.getopt(sys.argv[1:],"s:t:m:b:l:",["suffix=","test_file=","model=","batch_size=","num_labels="])
for o,a in opts:
    if o in ("-s","--suffix"):
        suffix = a
    if o in ("-t","--test_file"):
        input_test_file = a
    if o in ("-m","--model"):
        model_name = a
    if o in ("-b","--batch_size"):
        input_batch_size = int(a)

    if o in ("-l","--num_labels"):
        num_labels = int(a)


body = model_name.find("body")>=0
dynamic = model_name.find("dynamic")>=0 or model_name.find("lstm")>=0

deepmodel_util.num_labels = num_labels#
deepmodel_definition.num_labels = num_labels#
deepmodel_definition.emb_dim    = 300
deepmodel_definition.hidden_dim = 200

deepmodel_util.max_length_title = 30
deepmodel_util.max_length_body  = 100

deepmodel_definition.max_length_title = 30
deepmodel_definition.max_length_body  = 100

deepmodel_definition.filter_num = 200
deepmodel_definition.dropout_rate=0.5

process_setting(low =False,old = True,stop = False)


#data_train = "{}Data/middle/train_{}.txt".format(prefix,suffix)
data_test = input_test_file #"{}Data/middle/test_{}.txt".format(prefix,suffix)


data_title_vocab    = "{}Data/ready/title_{}.wl".format(prefix,suffix)
data_industry_vocab = "{}Data/ready/industry_{}.wl".format(prefix,suffix)
data_body_vocab = "{}Data/ready/body_{}.wl".format(prefix,suffix)

title_dict =     { x:i for i,x in enumerate([x.strip("\n") for x in open(data_title_vocab).readlines()])}
industry_dict =  { x:i for i,x in enumerate([x.strip("\n") for x in open(data_industry_vocab).readlines()])}

body_dict = {x:i for i,x in enumerate([x.strip("\n") for x in open(data_industry_vocab).readlines()])}




    
if dynamic:
    
    deepmodel_util.input_xt = C.input_variable(**Sequence[Tensor[1]])    
    deepmodel_util.test_data  = load_data_dynamic(data_test,title_dict,industry_dict)
    
elif body:
    
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])
    deepmodel_util.input_xb = C.input_variable(**Tensor[deepmodel_util.max_length_body])    
    deepmodel_util.test_data  = load_data_body(data_test,title_dict,body_dict,industry_dict)
    
else:   
    
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])  
    deepmodel_util.test_data  = load_data_static(data_test,title_dict,industry_dict)

#deepmodel_util.input_xt = C.input_variable(shape=(deepmodel_util.max_length_title))
deepmodel_util.input_y  = C.input_variable(shape=(1))

deepmodel_definition.input_xt_one_hot = C.one_hot(deepmodel_util.input_xt, num_classes=len(title_dict)   ,  sparse_output=True)
deepmodel_definition.input_y_one_hot = C.one_hot(deepmodel_util.input_y  , num_classes=len(industry_dict) ,  sparse_output=True)

deepmodel_util.input_xt_one_hot = deepmodel_definition.input_xt_one_hot
deepmodel_util.input_y_one_hot = deepmodel_definition.input_y_one_hot


#print(num_labels)

process_setting(low =False,old = True,stop = False)
deepmodel_util.num_labels=deepmodel_util.num_labels+1

model = load_model(model_name)#"model/180days_all_shuffled/lstm_acc0.858.dnn"

if body:
    test_body(input_batch_size,model,deepmodel_util.test_data)
else:
    test(input_batch_size,model,deepmodel_util.test_data)

print("Finished Test.")
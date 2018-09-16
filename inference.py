from cntk import load_model

import sys,getopt
from deepmodel_util import *
from deepmodel_definition import *
import deepmodel_util
import deepmodel_definition

model_name =""
suffix_file = ""
prefix=""
industry_file = ""
input_inference_file = ""
output_inference_file = ""

num_labels = 19
#input_lr = 5e-4
input_batch_size = 100
do_test=False
deepmodel_util.batch_size =1
opts,args = getopt.getopt(sys.argv[1:],"m:s:c:i:o:b:",["model=","suffix=","industry=","input=","output=","batch_size="])
for o,a in opts:
    if o in ("-s","--suffix"):
        suffix_file = a
    if o in ("-m","--model"):
        model_name = a
    if o in ("-c","--industry"):
        industry_file = a
    if o in ("-i","--input"):
        input_inference_file = a
    if o in ("-o","--output"):
        output_inference_file = a
    if o in ("-b","--batch_size"):
        deepmodel_util.batch_size = int(a)   
if industry_file == "":
    industry_file= suffix_file

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
    

#print("here",model_nam)
model = load_model(model_name)

data_industry_vocab = "{}Data/ready/industry_{}.wl".format(prefix,industry_file)
data_title_vocab    = "{}Data/ready/title_{}.wl".format(prefix,suffix_file)
data_body_vocab     = "{}Data/ready/body_{}.wl".format(prefix,suffix_file)

title_dict =     { x:i for i,x in enumerate([x.strip("\n") for x in open(data_title_vocab).readlines()])}
industry_dict =  { x:i for i,x in enumerate([x.strip("\n") for x in open(data_industry_vocab).readlines()])}
if body:
    body_dict =     { x:i for i,x in enumerate([x.strip("\n") for x in open(data_body_vocab).readlines()])}
    
if dynamic:
    
    deepmodel_util.input_xt = C.input_variable(**Sequence[Tensor[1]])      
elif body:
    
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])
    deepmodel_util.input_xb = C.input_variable(**Tensor[deepmodel_util.max_length_body])       
else:     
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])  

deepmodel_util.input_y  = C.input_variable(shape=(1))

deepmodel_definition.input_xt_one_hot = C.one_hot(deepmodel_util.input_xt, num_classes=len(title_dict)   ,  sparse_output=True)
deepmodel_definition.input_y_one_hot = C.one_hot(deepmodel_util.input_y  , num_classes=len(industry_dict) ,  sparse_output=True)

deepmodel_util.input_xt_one_hot = deepmodel_definition.input_xt_one_hot
deepmodel_util.input_y_one_hot = deepmodel_definition.input_y_one_hot

if body:
    inference_body(model,input_inference_file,output_inference_file,title_dict,body_dict,data_industry_vocab)
else:
    inference(model,input_inference_file,output_inference_file,title_dict,data_industry_vocab,dynamic=dynamic)
    
print("Inference Done.")

#inference(model,"Data/middle/1day_measure_sample_valid.txt","val/dyn_cnn_1day_measure_{}.txt".format(suffix),title_dict,data_industry_vocab,dynamic=True)

import sys,getopt
from deepmodel_util import *
from deepmodel_definition import *
import deepmodel_util
import deepmodel_definition

model  = "cnn"
suffix = ""
tag = "cnn"
prefix = ""
num_labels = 19
input_lr = 5e-4
input_epochs = 10
do_test=False
opts,args = getopt.getopt(sys.argv[1:],"s:m:l:t:b:",[
    "suffix=","model=","lr=","do_test=","epochs=","tag=","prefix=","batch_size=","num_labels=","--embed","--embed_title=","--embed_body="
])

for o,a in opts:
    if o in ("-s","--suffix"):
        suffix = a
    if o in ("-m","--model"):
        model_name = a
    if o == "--epochs":
        input_epochs = int(a)
    if o in ("l","--lr"):
        input_lr = float(a)
    if o == "--tag":
        tag = a
    if o == "--prefix":
        prefix = a
    if o == "--num_labels":
        num_labels = int(a)
    if o in ("--do_test","-t"):
        do_test=True
    if o in ("b","--batch_size"):
        batch_size=a
    

body = model.find("body")>=0
dynamic = model.find("dynamic")>=0

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


prefix = ""

data_train = "{}Data/middle/train_{}.txt".format(prefix,suffix)
data_test = "{}Data/middle/test_{}.txt".format(prefix,suffix)


data_title_vocab    = "{}Data/ready/title_{}.wl".format(prefix,suffix)
data_industry_vocab = "{}Data/ready/industry_{}.wl".format(prefix,suffix)
data_body_vocab = "{}Data/ready/body_{}.wl".format(prefix,suffix)

title_dict =     { x:i for i,x in enumerate([x.strip("\n") for x in open(data_title_vocab).readlines()])}
industry_dict =  { x:i for i,x in enumerate([x.strip("\n") for x in open(data_industry_vocab).readlines()])}

body_dict = {x:i for i,x in enumerate([x.strip("\n") for x in open(data_industry_vocab).readlines()])}


    
if dynamic:
    print("Dynamic")
    deepmodel_util.input_xt = C.input_variable(**Sequence[Tensor[1]])
    if do_test:
        deepmodel_util.test_data  = load_data_dynamic(data_test,title_dict,industry_dict)
    deepmodel_util.train_data = load_data_dynamic(data_train,title_dict,industry_dict)
elif body:
    print("Body")
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])
    deepmodel_util.input_xb = C.input_variable(**Tensor[deepmodel_util.max_length_body])
    if do_test:
        deepmodel_util.test_data  = load_data_body(data_test,title_dict,industry_dict)
    deepmodel_util.train_data = load_data_body(data_train,title_dict,industry_dict)
else:   
    deepmodel_util.input_xt = C.input_variable(**Tensor[deepmodel_util.max_length_title])
    if do_test:
        deepmodel_util.test_data  = load_data_static(data_test,title_dict,industry_dict)
    deepmodel_util.train_data = load_data_static(data_train,title_dict,industry_dict)

deepmodel_util.input_y  = C.input_variable(shape=(1))

deepmodel_definition.input_xt_one_hot = C.one_hot(deepmodel_util.input_xt, num_classes=len(title_dict)   ,  sparse_output=True)
deepmodel_definition.input_y_one_hot = C.one_hot(deepmodel_util.input_y  , num_classes=len(industry_dict) ,  sparse_output=True)
deepmodel_util.input_xt_one_hot = deepmodel_definition.input_xt_one_hot
deepmodel_util.input_y_one_hot = deepmodel_definition.input_y_one_hot
if embed:
    if not body:
        deepmodel_definition.embedding = load_embedding(data_title_vocab,embed_title)
    else:
        deepmodel_definition.embedding_title = load_embedding(data_title_vocab,embed_title)
        deepmodel_definition.embedding_body = load_embedding(data_body_vocab,embed_body)
    

if body:
    train_body(create_model(model)(),deepmodel_util.train_data,num_epochs=input_epochs,learning_rate=input_lr,batch_size = batch_size,tag =tag,do_test = do_test)
else:
    train(create_model(model)(),deepmodel_util.train_data,num_epochs=input_epochs,learning_rate=input_lr,batch_size = batch_size,tag = tag,do_test=do_test)
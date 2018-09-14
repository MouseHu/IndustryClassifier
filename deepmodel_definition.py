import cntk as C
def get_context_left(current,previous,w_l,w_ls):
    left_c = current@w_l
    left_e = previous@w_ls
    left_h=left_c+left_e
    return C.relu(left_h)
def get_context_right(current,after,w_r,w_rs):
    right_c = current@w_r
    right_e = after@w_rs
    right_h =right_c+right_e
    return C.relu(right_h)

def create_model_rcnn(input_one_hot,max_length,embed = False,embedding=None):
    first_word = C.parameter(shape=(emb_dim))
    last_word = C.parameter(shape=(emb_dim))
    w_l,w_ls,w_r,w_rs = C.parameter(shape=(emb_dim,emb_dim)),C.parameter(shape=(emb_dim,emb_dim)),C.parameter(shape=(emb_dim,emb_dim)),C.parameter(shape=(emb_dim,emb_dim))
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    if embed:
        h1= C.layers.Embedding(weights=embedding,name='embed_1')(input_one_hot)#
    else:
        h1= C.layers.Embedding(emb_dim,name='embed_2')(input_one_hot)#init=embedding,
    previous = first_word
    # h1 [batch*sentence_length*emb_dim]
    context_left_list = []
    for i in range(max_length):
        current = C.squeeze(h1[i])
        context_left_list.append(get_context_left(current,previous,w_l,w_ls))
        previous = current
        
    context_right_list = []
    after = last_word
    for i in reversed(range(max_length)):
        current = C.squeeze(h1[i])
        context_right_list.append(get_context_right(current,after,w_r,w_rs))
        after = current
    total_list = []
    for i in range(max_length_title):
        total_list.append(C.splice(h1[i],context_left_list[i],context_right_list[i]))
    h3=C.element_max(*total_list)
    return h3
def create_model_rcnn_normal():
    h3 = create_model_rcnn(input_xt_one_hot,max_length_title,embed=False)
    drop1 = C.layers.Dropout(dropout_rate)(h3)
    h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4
def create_model_rcnn_body_2fold():
    h3_static_title = create_model_rcnn(input_xt_one_hot,max_length_title,embed=True,embedding=embedding_title)
    h3_dynamic_title = create_model_rcnn(input_xt_one_hot,max_length_title,embed=False)
    h3_static_body = create_model_rcnn(input_xb_one_hot,max_length_body,embed=True,embedding=embedding_body)
    h3_dynamic_body = create_model_rcnn(input_xb_one_hot,max_length_body,embed=False)
    h3 = C.splice(h3_static_title,h3_dynamic_title,h3_static_body,h3_dynamic_body)
    drop1 = C.layers.Dropout(dropout_rate)(h3)
    h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4

def create_model_rcnn_body():
    #h3_static_title = create_model_rcnn(input_xt_one_hot,max_length_title,embed=True,embedding=embedding_title)
    h3_dynamic_title = create_model_rcnn(input_xt_one_hot,max_length_title,embed=False)
    #h3_static_body = create_model_rcnn(input_xb_one_hot,max_length_body,embed=True,embedding=embedding_body)
    h3_dynamic_body = create_model_rcnn(input_xb_one_hot,max_length_body,embed=False)
    h3 = C.splice(h3_dynamic_title,h3_dynamic_body)
    drop1 = C.layers.Dropout(dropout_rate)(h3)
    h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4

def create_model_cnn(embed = False):
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    if embed:
        h1= C.layers.Embedding(weights=embedding,name='embed_1')(input_xt_one_hot)#
    else:
        h1= C.layers.Embedding(emb_dim,name='embed_2')(input_xt_one_hot)#init=embedding,

    

    h2_1=C.layers.Convolution((1,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(h1)
    h2_2=C.layers.Convolution((2,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(h1)
    h2_3=C.layers.Convolution((3,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu)(h1)
    
    h3_1=C.layers.MaxPooling((max_length_title-0,1),name='pooling_1')(h2_1)
    h3_2=C.layers.MaxPooling((max_length_title-1,1),name='pooling_2')(h2_2)
    h3_3=C.layers.MaxPooling((max_length_title-2,1),name='pooling_3')(h2_3)
    #h2=BiRecurrence(C.layers.LSTM(hidden_dim), C.layers.LSTM(hidden_dim))(h1)
    h3=C.splice(h3_2,h3_1,h3_3,axis=0)
    drop1 = C.layers.Dropout(dropout_rate)(h3)
    h4=C.layers.Dense(300,name='hidden')(drop1)
    h5=C.layers.Dense(num_labels,name='hidden')(h4)
 
    return h5

def create_model_rcnn_with_cnn():
    logit1 = create_model_cnn()
    logit2 = create_model_rcnn()
    weight1 = C.parameter(shape=(1),init=0.5)
    weight2 = 1-weight1
    logit = weight1*logit1+weight2*logit2
    return logit 

def create_model_cnn_dynamic(embed = False):
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    if embed:
        h1= C.layers.Embedding(weights=embedding,name='embed_1')(input_xt_one_hot)#
    else:
        h1= C.layers.Embedding(emb_dim,name='embed_2')(input_xt_one_hot)#init=embedding,

    h1 = C.squeeze(h1)
    print(h1)
    h2_1=C.layers.Convolution((1,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu,sequential=True)(h1)
    h2_2=C.layers.Convolution((2,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu,sequential=True)(h1)
    h2_3=C.layers.Convolution((3,emb_dim),num_filters=filter_num,reduction_rank=0,activation=C.relu,sequential=True)(h1)
    seq_MaxPooling = C.layers.Fold(C.element_max)
    h3_1=seq_MaxPooling(h2_1)
    h3_2=seq_MaxPooling(h2_2)
    h3_3=seq_MaxPooling(h2_3)
    #h2=BiRecurrence(C.layers.LSTM(hidden_dim), C.layers.LSTM(hidden_dim))(h1)
    h3=C.splice(h3_2,h3_1,h3_3,axis=0)
    drop1 = C.layers.Dropout(dropout_rate)(h3)
    h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4

def create_model_cnn_2fold(dynamic = False):
    #version 2 : 1 dense layer version3: sigmoid activation in dense
    #
    with C.layers.default_options(initial_state=0.1):


        h1_1= C.layers.Embedding(weights=embedding,name='embed_1')(input_xt_one_hot)#
        h1_2= C.layers.Embedding(300,name='embed_2')(input_xt_one_hot)#init=embedding,
        
        
        
        h1_1_expand = C.expand_dims(h1_1,-3)
        h1_2_expand = C.expand_dims(h1_2,-3)
        
        h1 = C.splice(h1_1_expand,h1_2_expand,axis = -3)
        
        #bn = C.layers.BatchNormalization(name='bn')(h1)
        

        #value,valid = to_static(h1)

        filter_num=100

        h2_1=C.layers.Convolution((3,emb_dim),num_filters=filter_num,reduction_rank=1,activation=C.relu)(h1)
        h2_2=C.layers.Convolution((4,emb_dim),num_filters=filter_num,reduction_rank=1,activation=C.relu)(h1)
        h2_3=C.layers.Convolution((5,emb_dim),num_filters=filter_num,reduction_rank=1,activation=C.relu)(h1)
        if dynamic:
            seq_MaxPooling = C.layers.Fold(C.element_max)
            h3_1=seq_MaxPooling(h2_1)
            h3_2=seq_MaxPooling(h2_2)
            h3_3=seq_MaxPooling(h2_3)
        else:
            h3_1=C.layers.MaxPooling((max_length_title-2,1),name='pooling_1')(h2_1)
            h3_2=C.layers.MaxPooling((max_length_title-3,1),name='pooling_2')(h2_2)
            h3_3=C.layers.MaxPooling((max_length_title-4,1),name='pooling_3')(h2_3)
        
        h3=C.splice(h3_2,h3_1,h3_3,axis=0)
        drop1 =C.layers.Dropout(0.5)(h3)
        h4=C.layers.Dense(num_labels,name='hidden')(drop1)

    return h4

def create_model_cnn_with_body():
    
    h1t= C.layers.Embedding(300,name='embed')(input_xt_one_hot)#init=embedding,
    h1b= C.layers.Embedding(300,name='embed')(input_xb_one_hot)#init=embedding,
    
    #bnb = C.layers.BatchNormalization(name='bn')(h1b)
    #bnt = C.layers.BatchNormalization(name='bn')(h1t)



    h2_1t=C.layers.Convolution((1,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1t)
    h2_2t=C.layers.Convolution((2,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1t)
    h2_3t=C.layers.Convolution((3,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1t)

    h2_1b=C.layers.Convolution((1,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1b)
    h2_2b=C.layers.Convolution((2,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1b)
    h2_3b=C.layers.Convolution((3,emb_dim),num_filters=50,reduction_rank=0,activation=C.relu)(h1b)

    h3_2t=C.layers.MaxPooling((max_length_title-1,1),name='pooling_t_1')(h2_2t)
    h3_1t=C.layers.MaxPooling((max_length_title-0,1),name='pooling_t_2')(h2_1t)
    h3_3t=C.layers.MaxPooling((max_length_title-2,1),name='pooling_t_3')(h2_3t)

    h3_2b=C.layers.MaxPooling((max_length_body-1,1),name='pooling_b_1')(h2_2b)
    h3_1b=C.layers.MaxPooling((max_length_body-0,1),name='pooling_b_2')(h2_1b)
    h3_3b=C.layers.MaxPooling((max_length_body-2,1),name='pooling_b_3')(h2_3b)

    h3=C.splice(h3_2t,h3_1t,h3_3t,h3_2b,h3_1b,h3_3b,axis=0)

    #h4=C.layers.Dense(hidden_dim, activation=C.relu,name='hidden')(h3)
    #drop1 = C.layers.Dropout(0.5,name='drop1')(h3)

    h4=C.layers.Dense(num_labels,name='classify')(h3)

    return h4
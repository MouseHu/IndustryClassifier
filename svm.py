#fetch newsgroupdata
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
from sklearn.svm import SVC,LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import random
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]
    
    
# evaluation code
num_labels=19
#industry=[i.rstrip("\n") for i in open("news//industry.wl").readlines()]
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def evaluate(predict,label,num_labels=19,save_file=""):
    
    predict=np.array(predict).astype(int)
    label=np.array(label).astype(int)
    confuse = fast_hist(predict,label,num_labels)
    precision=np.diag(confuse)/np.sum(confuse,axis=1)
    recall = np.diag(confuse)/np.sum(confuse,axis=0)
    accuarcy = np.diag(confuse).sum() / confuse.sum()
    aver_precision=np.nanmean(precision)
    aver_recall = np.nanmean(recall)
    #print(confuse.astype(int))
    if save_file != "":
        np.savetxt(save_file, confuse.astype(int), delimiter=',', fmt='%s')
    print("Recall:")
    print("\n".join([category[i]+"\t\t"+str(recall[i]) for i in range(num_labels)]))
    print("\nPrecision:")
    print("\n".join([category[i]+"\t\t"+str(precision[i]) for i in range(num_labels)]))
    #print([(category[i],precision[i]) for i in range(num_labels)])
    print("\nF-Score:")
    print("\n".join([category[i]+"\t\t"+str(2/(1/precision[i]+1/recall[i])) for i in range(num_labels)]))
    print(aver_precision,aver_recall,accuarcy)
    
def pr_plot(predict,label,predict_prob,num_labels=19):
    threshold = np.linspace(0,0.9,10)
    precision = []
    recall = []
    for th in threshold:
        my_predict = np.copy(predict)
        my_predict[predict_prob<th]=num_labels
        confuse = fast_hist(my_predict,label,num_labels+1)
        precision_th=np.diag(confuse)/np.sum(confuse,axis=1)
        recall_th = np.diag(confuse)/np.sum(confuse,axis=0)
        precision.append(np.nanmean(precision_th[:-1]))
        recall.append(np.nanmean(recall_th[:-1]))
    p, = plt.plot(threshold,precision,'b')
    #plt.legend("precision")
    r, =plt.plot(threshold,recall,'r')
    plt.legend([p,r],["precision","recall"])
    plt.show()
    plt.plot(precision,recall)
    plt.show()
    
def load_data(input_file,industry_file,with_body=False):
    global industry,category
    global num_labels
    industry = {x.rstrip("\n"):i for i,x in enumerate(open(industry_file,"r",encoding = "utf-8").readlines())}
    category = [x.rstrip("\n") for x in open(industry_file,"r",encoding = "utf-8").readlines()]
    num_labels = len(category)
    
    title = [x.split("\t")[0] for x in open(input_file,"r",encoding = "utf-8").readlines()]
    label=[industry[x.split("\t")[1].rstrip("\n")] for x in open(input_file,"r",encoding = "utf-8").readlines()]
    if with_body:
        body = [x.split("\t")[2] for x in open(input_file,"r",encoding = "utf-8").readlines()]
        doc = {'title':title,'body':body}
        return doc,label
    else:
        return title,label
        
def load_data_newsgroup():
    global doc,label,test_doc,test_label
    global num_labels
    global industry,category
    #newsgroup dataset
    twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
    doc = twenty_train.data
    label = twenty_train.target

    test_doc = twenty_test.data
    test_label=twenty_test.target
    
    category = list(twenty_train.target_names)
    num_labels=len(category)
    return doc,label,test_doc,test_label
    
def val_predict(text_clf,val_file,result_file):
    global val_doc
    val_doc = [x.rstrip("\n") for x in open(val_file,"r",encoding='utf-8').readlines()]
    val_predict=text_clf.predict(val_doc)
    result = open(result_file,"w",encoding="utf-8")
    for i in val_predict:
        print(category[i])
        result.write(category[i]+"\n")
    result.close()
    
    
def train(text_clf,title,label,label_num=num_labels):
    
    text_clf = text_clf.fit(title,label)
    print("Model Training Finished.")
    
def test(text_clf,test_title,test_label,label_num=num_labels):
    predicted = text_clf.predict(test_title)
    
    print(np.mean(predicted == test_label))
    evaluate(predicted,test_label,label_num)
    try:
        predicted_prob = text_clf.predict_proba(test_title)
        predicted_prob = np.argmax(predicted_prob,axis=1)
        pr_plot(predicted,predicted_prob,test_label,label_num)
    except:
        pass
    

def grid_search(text_clf_svm,parameter,doc,label):
    gs_clf = GridSearchCV(text_clf_svm, parameter, n_jobs=-1)
    
    gs_clf = gs_clf.fit(doc, label)
    
    print(gs_clf.best_score_)
    print(gs_clf.best_params_)
    
def val(gt,predict,industry,predict_prob= None):
    industry ={x.rstrip("\n"):i for i,x in enumerate(open(industry,encoding = "utf-8").readlines())}
    gt =[industry[x.rstrip("\n")] for x in open(gt,encoding = "utf-8").readlines()]
    predict =[industry[y] if y in industry else len(industry) for y in [x.rstrip("\n").lower().replace(" ","")  for x in open(predict,encoding = "utf-8").readlines()]]
    gt = np.array(gt)
    predict = np.array(predict)
    confuse = fast_hist(gt,predict,len(industry)+1)
    evaluate(predict,gt,len(industry))
    if predict_prob is not None:
        prob =[float(x.rstrip("\n")) for x in open(predict_prob,encoding = "utf-8").readlines()]
        pr_plot(predict,gt,prob,len(industry))
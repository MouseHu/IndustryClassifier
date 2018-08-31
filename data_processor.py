import re
import numpy as np
from cucco import Cucco
import nltk
cucco = Cucco()

stop_word=["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", 
           "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", 
           "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", 
           "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", 
           "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", 
           "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", 
           "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", 
           "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", 
           "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", 
           "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", 
           "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", 
           "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
           "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed",
           "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less",
           "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", 
           "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", 
           "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", 
           "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", 
           "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same",
           "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", 
           "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime",
           "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their",
           "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein",
           "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through",
           "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty",
           "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what",
           "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein",
           "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom",
           "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself",
           "yourselves", "the"]


pre_low =False
pre_old = False
pre_stop = True
pre_random = False

def clean_str(string):
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.replace("  "," ")
    return string.strip()
def split(title):
    if pre_low:
        title = title.lower()
    title=title.replace(u"\u2029","").replace(u'\xa0', u' ')
    title=title.replace("’","'")
    title=title.replace("‘","'")
    title=title.replace("'m'"," am")
    title=title.replace("'s"," is")
    title=title.replace("'re"," are")
    title=title.replace("'ll"," will")
    raw= clean_str(title).split(" ")
    if pre_stop:
        return [x for x in raw if x.lower() not in stop_word]
    else:
        return raw
def compose_random(word_list):
    if len(word_list)==0:
        print("hei!")
        return ""
    my_list=copy.copy(word_list)
    a=np.random.randint(4)
    b=np.random.randint(len(word_list))
    c=np.random.randint(len(word_list))
    if a ==1:
        del my_list[b]
    if a == 2:
        my_list.insert(b,'UNK')
        #print(my_list)
    if a == 3:
        tmp=my_list[b]
        my_list[b]=my_list[c]
        my_list[c]=tmp
    return compose(my_list)

def compose(word_list):
    return (" ".join(word_list)).replace(u'\xa0', u' ').strip(" ")


#def process_new(title):
#    title = cucco.normalize(title)
#    title = re.sub(r"\s‘", " ‘ ", title)
#    title = re.sub(r"’\s", " ’ ", title)
#    title = re.sub(r"\s“", " “ ", title)
#    title = re.sub(r"”\s", " ” ", title)
    #title = re.sub(r"\A‘", " ‘ ", title)
    #title = re.sub(r"’\Z", " ’ ", title)
    #title = re.sub(r"\A“", " “ ", title)
    #title = re.sub(r"”\Z", " ” ", title)
    
    #title = re.sub(r"\s\s", " ", title)
    #title = re.sub(r"([^\s])’s", r"\1 ’s", title)
    #title = re.sub(r"([^\s])’re", r"\1 ’re", title)
    #title = re.sub(r"([^\s])’ll", r"\1 ’ll", title)
    #title = re.sub(r"n’t", " not", title)
    
#    title = re.sub(r"[0-9]+[A-Z,a-z]\s", "NUMBER ", title)
#    title = re.sub(r"[0-9]+", " NUMBER ", title)#
#    title = re.sub(r"\s\s", " ", title)
    
#    title = re.sub(r"([^A-Z,a-z,\",'])", r" \1 ", title)
    
#    title = re.sub(r"\s\s", " ", title)
#    title = re.sub(r"\s\s", " ", title)
#    title = re.sub(r"\A\s", "", title)
#    title = re.sub(r"\s\Z", "", title)
#    return title
def process_new(sentence):
    return " ".join(nltk.word_tokenize(sentence))
def process_property():
    return {'low':pre_low,'old':pre_old,'stop':pre_stop}
def process_setting(low =True,old = True,stop = True):
    global pre_low,pre_old,pre_stop
    pre_low=low
    pre_old = old
    pre_stop = stop
def tokenize(title):
    if pre_old:
        if pre_random:
            return compose_random(split(title))
        else:
            return compose(split(title))
    else:
        return process_new(title)
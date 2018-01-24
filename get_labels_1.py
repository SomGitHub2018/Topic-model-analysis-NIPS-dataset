import gensim
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
import math
from collections import defaultdict
from numpy import exp, log, dot, zeros, outer, random, dtype, float32 as REAL,\
    double, uint32, seterr, array, uint8, vstack, fromstring, sqrt, newaxis,\
ndarray, empty, sum as np_sum, prod, ones, ascontiguousarray
from gensim import utils,matutils
import re
import pickle

data ="toytopics.csv"

doc2vecmodel = "pre_trained_models/doc2vec/docvecmodel.d2v"
word2vecmodel = "pre_trained_models/word2vec/word2vec" 
num_candidates =19
output_filename = "output_candidates"
doc2vec_indices = "support_files/doc2vec_indices" 
word2vec_indices = "support_files/word2vec_indices"


with open(doc2vec_indices,'rb') as m:   
    d_indices=pickle.load(m)
with open(word2vec_indices,'rb') as n:
    w_indices =pickle.load(n)

model1 =Doc2Vec.load(doc2vecmodel)
model2 = Word2Vec.load(word2vecmodel)
print "models loaded"

topics = pd.read_csv(data)
try:
    new_frame= topics.drop('domain',1)
    topic_list = new_frame.set_index('topic_id').T.to_dict('list')
except:
    topic_list = topics.set_index('topic_id').T.to_dict('list')
print "Data Gathered"


w_indices = list(set(w_indices))
d_indices = list(set(d_indices))

model1.syn0norm = (model1.syn0 / sqrt((model1.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model1.docvecs.doctag_syn0norm =  (model1.docvecs.doctag_syn0 / sqrt((model1.docvecs.doctag_syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)[d_indices]
print "doc2vec normalized"

model2.syn0norm = (model2.syn0 / sqrt((model2.syn0 ** 2).sum(-1))[..., newaxis]).astype(REAL)
model3 = model2.syn0norm[w_indices]
print "word2vec normalized"

def get_word(word):
    if type(word)!=str:
        return word
    inst = re.search(r"_\(([A-Za-z0-9_]+)\)", word)
    if inst == None:
        return word
    else:
        word = re.sub(r'_\(.+\)','',word)
        return word

def get_labels(topic_num):
    valdoc2vec =0.0
    valword2vec =0.0
    cnt = 0
    store_indices =[]
    
    print "Processing Topic number " +str(topic_num)
    for item in topic_list[topic_num]:
        try: 
            tempdoc2vec = model1.syn0norm[model1.vocab[item].index] # The word2vec value of topic word from doc2vec trained model
        except:
            pass
        else:
            meandoc2vec = matutils.unitvec(tempdoc2vec).astype(REAL)    # Getting the unit vector
            distsdoc2vec = dot(model1.docvecs.doctag_syn0norm, meandoc2vec) # The dot product of all labels in doc2vec with the unit vector of topic word
            valdoc2vec = valdoc2vec + distsdoc2vec

        try:
            tempword2vec = model2.syn0norm[model2.vocab[item].index]  # The word2vec value of topic word from word2vec trained model
        except:
            pass
        else:
            meanword2vec = matutils.unitvec(tempword2vec).astype(REAL) # Unit vector 

            distsword2vec = dot(model3, meanword2vec) # The dot prodiuct of all possible labels in word2vec vocab with the unit vector of topic word

              
            if (model2.vocab[item].index) in w_indices:
                
                i_val = w_indices.index(model2.vocab[item].index)
      		store_indices.append(i_val)
                distsword2vec[i_val] =0.0
            valword2vec = valword2vec + distsword2vec
    
    avgdoc2vec = valdoc2vec/float(len(topic_list[topic_num])) # Give the average vector over all topic words
    avgword2vec = valword2vec/float(len(topic_list[topic_num])) # Average of word2vec vector over all topic words

    bestdoc2vec = matutils.argsort(avgdoc2vec, topn = 100, reverse=True) # argsort and get top 100 doc2vec label indices 
    resultdoc2vec =[]
    for elem in bestdoc2vec:
        ind = d_indices[elem]
        temp = model1.docvecs.index_to_doctag(ind)
        resultdoc2vec.append((temp,float(avgdoc2vec[elem])))
   
    for element in store_indices:
        avgword2vec[element] = (avgword2vec[element]*len(topic_list[topic_num]))/(float(len(topic_list[topic_num])-1))
    
    bestword2vec = matutils.argsort(avgword2vec,topn=100, reverse=True) #argsort and get top 100 word2vec label indices
    # Get the word2vec labels from indices
    resultword2vec =[]
    for element in bestword2vec:
        ind = w_indices[element]
        temp = model2.index2word[ind]
        resultword2vec.append((temp,float(avgword2vec[element])))
    
    comb_labels = list(set([i[0] for i in resultdoc2vec]+[i[0] for i in resultword2vec]))
    newlist_doc2vec = []
    newlist_word2vec =[]

    for elem in comb_labels:
        try:
            
            newlist_doc2vec.append(d_indices.index(model1.docvecs.doctags[elem].offset))
            temp = get_word(elem)
            newlist_word2vec.append(w_indices.index(model2.vocab[temp].index))
            
        except:
            pass
    newlist_doc2vec = list(set(newlist_doc2vec))
    newlist_word2vec = list(set(newlist_word2vec))

    resultlist_doc2vecnew =[(model1.docvecs.index_to_doctag(d_indices[elem]),float(avgdoc2vec[elem])) for elem in newlist_doc2vec]
    resultlist_word2vecnew =[(model2.index2word[w_indices[elem]],float(avgword2vec[elem])) for elem in newlist_word2vec]
    new_score =[]
    for item in resultlist_word2vecnew:
        k,v =item
        for elem in resultlist_doc2vecnew:
            k2,v2 = elem
            k3 =get_word(k2)
            if k==k3:
                v3 = v+v2
                new_score.append((k2,v3))
    new_score = sorted(new_score,key =lambda x:x[1], reverse =True)
    return new_score[:(int(num_candidates))]


result=[]
for i in range(0,len(topic_list)):
	result.append(get_labels(i))
g = open(output_filename,'w')
for i,elem in enumerate(result):
    val = ""
    for item in elem:
        val = val + " "+item[0]
    g.write(val+"\n")
g.close()

print "Candidate labels written to "+output_filename
print "\n"

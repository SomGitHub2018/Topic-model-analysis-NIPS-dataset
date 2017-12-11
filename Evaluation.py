import pickle
import numpy as np
import gensim
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models

#load pickle
pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
pickle_in.close()

df9_yrs = p_dict['df9_yrs']
distinct_yrs = p_dict['distinct_yrs']
corpus = p_dict['corpus']


print("Reading data")
# df9 = pd.read_csv("papers.csv")
df9 = pd.read_csv("../data/papers.csv", encoding="utf8", engine="python")


tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Load model
model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')

def getCorpus(x):
    df9_yrs = df9['year']
    distinct_yrs = sorted(set(df9_yrs))
    p_dict['df9_yrs'] = df9_yrs
    p_dict['distinct_yrs'] = distinct_yrs

    texts = []
    for i in x:
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2)]

        # stem tokens
        lemmatized_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(lemmatized_tokens)

    df9['Cleaned_PaperText'] = pd.Series(texts, index=df9.index)

    print("Creating dictionary")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus


# divide text to two halves, we get two corpuses
def data_preprocessing():
    print("Data pre processing")
    ################################### Remove numbers and single letter words
    # loop through document list
    i1_lst = []
    i2_lst = []
    for i in df9['paper_text']:
        # clean and tokenize document string
        i_1, i_2 = i[:int(len(i)/2)], i[int(len(i)/2):]

        i1_lst.append(i_1)
        i2_lst.append(i_2)

    corpus_fh = getCorpus(i1_lst) #first half
    corpus_sh = getCorpus(i2_lst) #second half

    p_dict['corpus_fh'] = corpus_fh
    p_dict['corpus_sh'] = corpus_sh


def choose_max_prob_topic():
    corpus_fh = p_dict['corpus_fh']
    corpus_sh = p_dict['corpus_sh']

    equals = 0
    notEquals = 0

    for i in range(len(corpus)):
        topic_prob = model[corpus_fh[i]]
        temp = []
        for a,b in topic_prob:
            temp.append(b)
        fh = np.argmax(temp)

        topic_prob = model[corpus_sh[i]]
        temp = []
        for a, b in topic_prob:
            temp.append(b)
        sh = np.argmax(temp)

        if(fh == sh):
            equals += 1
        else:
            notEquals += 1

    print(equals)
    print(notEquals)




#Main Fn Call
# data_preprocessing()
choose_max_prob_topic()

pickle_out = open("pickle_data.pickle", "wb")
pickle.dump(p_dict, pickle_out)
pickle_out.close()

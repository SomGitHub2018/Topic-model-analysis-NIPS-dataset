import pandas as pd
# import csv
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import gensim
import pyLDAvis
import pickle

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()



pickle_out = open("pickle_data.pickle","wb")
p_dict = {} # pickle dictionary

def pyladavis_viz():
    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda.html')

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

print("Reading data")
df9 = pd.read_csv("../data/papers.csv", encoding="utf8", engine="python")
df9_yrs = df9['year']
distinct_yrs = sorted(set(df9_yrs))
p_dict['df9_yrs'] = df9_yrs
p_dict['distinct_yrs'] = distinct_yrs

texts = []

print("Data pre processing")
################################### Remove numbers and single letter words
# loop through document list
for i in df9['paper_text']:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2)]

    # lemmatize tokens
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(lemmatized_tokens)

df9['Cleaned_PaperText'] = pd.Series(texts, index=df9.index)

print("Creating dictionary")
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
df9['Corpus'] = pd.Series(corpus, index=df9.index)
p_dict['dictionary'] = dictionary
p_dict['corpus'] = corpus


# saving pickle file
pickle.dump(p_dict, pickle_out)
pickle_out.close()

print("Creating model")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
print(ldamodel.print_topics(num_topics=10, num_words=4))

print("Saving model")
model = ldamodel
# save model
ldamodel.save('output/model.atmodel')
print("model saved")

# Load model
# model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')


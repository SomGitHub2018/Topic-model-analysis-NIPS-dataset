import pickle
import gensim
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

#load pickle
pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
pickle_in.close()

def pyladavis_viz():
    # Load model
    model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')
    corpus = p_dict['corpus']
    dictionary = p_dict['dictionary']

    vis = pyLDAvis.gensim.prepare(model, corpus, dictionary)
    pyLDAvis.save_html(vis, 'lda.html')


pyladavis_viz()

tokenizer = RegexpTokenizer(r'\w+')
en_stop = set(stopwords.words('english'))
p_stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

print("Loading data")
nips = pd.read_csv("../data/papers.csv", encoding="utf8", engine="python")
nips_yrs = nips['year']
distinct_yrs = sorted(set(nips_yrs))

texts = []

print("Preprocessing data")
# iterate each document
for i in nips['paper_text']:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # stop words removed
    stopped_tokens = [i for i in tokens if (not i in en_stop and not str(i).isdigit() and len(str(i)) > 2)]

    # lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(i) for i in stopped_tokens]

    texts.append(lemmatized_tokens)

nips['Cleaned_PaperText'] = pd.Series(texts, index=nips.index)

print("Creating the Dictionary")
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
nips['Corpus'] = pd.Series(corpus, index=nips.index)

# saving pickle file
p_dict = {} # pickle dictionary
p_dict['dictionary'] = dictionary
p_dict['corpus'] = corpus
p_dict['df9_yrs'] = nips_yrs
p_dict['distinct_yrs'] = distinct_yrs
pickle_out = open("pickle_data.pickle","wb")
pickle.dump(p_dict, pickle_out)
pickle_out.close()


print("Creating the model")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=20)
print(ldamodel.print_topics(num_topics=10, num_words=4))

print("Saving the model")
model = ldamodel
ldamodel.save('output/model.atmodel')
print("Program complete")


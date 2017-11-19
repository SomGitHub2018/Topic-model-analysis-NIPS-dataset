
import gensim

# Load model
model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')
x = 2


for i in model.print_topics(num_topics=10, num_words=4):
    print(i)
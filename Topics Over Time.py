import gensim
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import spline

#load pickle
pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
pickle_in.close()


# returns
# topic_yr_map = {1987: [23.477306301575261, 17.74636315217273, 26.847394613504079, 11.997810091403467, 6.4015614721370984,
# doc_yr_count = {1987: 90, 1988: 94, 1989: 101, 1990: 143, 1991: 144, 1992: 127, 1993: 158,
def avg_topic_probability():
    for i in range(len(corpus)):
        text = corpus[i]
        yr = df9_yrs[i]

        topic_prob = model[corpus[i]]
        doc_yr_count[yr] += 1

        cnt = 0
        for x in topic_prob:
            topic_yr_map[yr][cnt] += x[1]
            cnt += 1
    return topic_yr_map, doc_yr_count

def choose_max_prob_topic():
    for i in range(len(corpus)):
        text = corpus[i]
        yr = df9_yrs[i]

        topic_prob = model[corpus[i]]
        doc_yr_count[yr] += 1

        max_tuple = max(topic_prob)
        topic_yr_map[yr][max_tuple[0]] += 1
    print(topic_yr_map)
    print(doc_yr_count)

    p_dict['topic_yr_map'] = topic_yr_map
    p_dict['doc_yr_count'] = doc_yr_count

    return topic_yr_map, doc_yr_count

# returns {1987: [2.6085895890639179, 1.9718181280191922, 2.983043845944898, 1.3330900101559409, 0.71128460801523308,...
def normalize_for_viz(x,y,yrs):
    for yr in yrs:
        temp = []
        for i in y[yr]:
            temp.append(i * 10 / x[yr])
        y[yr] = temp
    return y

#a = topic_yr_map
#x = distinct_yrs
def viz_tot(a, x): # a = yr_topic_map
    a_vals = list(a.values())
    y = list(zip(*a_vals))

    xnew = np.linspace(min(x),max(x),200) #300 represents number of points to make between T.min and T.max

    for i in range(0,10):
        power_smooth = spline(x,y[i],xnew)
        plt.plot(xnew,power_smooth, label="Topic"+str(i))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # plt.savefig("/images/TOT.png", dpi=500)
    plt.show()


df9_yrs = p_dict['df9_yrs']
distinct_yrs = p_dict['distinct_yrs']
corpus = p_dict['corpus']

# Load model
model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')

# Initialization : topic prob distribution for each year
# 1987:[0....0], 1988:[0,..0],
topic_yr_map = {}
doc_yr_count = {}
for i in distinct_yrs:
    # topic_yr_map[i] = [0]*len(distinct_yrs)
    topic_yr_map[i] = [0]*10
    doc_yr_count[i] = 0


# avg_topic_probability() # TOT - 1
choose_max_prob_topic() # TOT - 2
topic_yr_map = normalize_for_viz(doc_yr_count, topic_yr_map, distinct_yrs)
viz_tot(topic_yr_map, distinct_yrs)


pickle_out = open("pickle_data.pickle", "wb")
pickle.dump(p_dict, pickle_out)
pickle_out.close()
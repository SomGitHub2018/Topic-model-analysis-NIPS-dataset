import gensim
import matplotlib.pyplot as plt
import pickle
import numpy as np
from scipy.interpolate import spline


def normalize_for_viz(x,y,yrs):
    for yr in yrs:
        temp = []
        for i in y[yr]:
            temp.append(i * 10 / x[yr])
        y[yr] = temp
    return y


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



pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
df9_yrs = p_dict['df9_yrs']
distinct_yrs = p_dict['distinct_yrs']
corpus = p_dict['corpus']

# Load model
model = gensim.models.ldamodel.LdaModel.load('output/model.atmodel')

# 1987:[0....0], 1988:[0,..0],
topic_yr_map = {}
doc_yr_count = {}
for i in distinct_yrs:
    topic_yr_map[i] = [0]*len(distinct_yrs)
    doc_yr_count[i] = 0


for i in range(len(corpus)):
    text = corpus[i]
    yr = df9_yrs[i]

    topic_prob = model[corpus[i]]
    doc_yr_count[yr] += 1

    cnt = 0
    for x in topic_prob:
        topic_yr_map[yr][cnt] += x[1]
        cnt += 1


topic_yr_map = normalize_for_viz(doc_yr_count, topic_yr_map, distinct_yrs)
viz_tot(topic_yr_map, distinct_yrs)
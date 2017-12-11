import numpy as np
import scipy
import scipy.interpolate
from scipy import *
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import spline
from scipy import interp, arange, exp

#load pickle
pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
pickle_in.close()

topic_number = 5
x = p_dict['distinct_yrs']
topic_yr_map = (p_dict['topic_yr_map'])
y = p_dict['topic_yr_vals'][topic_number]
pred_yrs = list(range(2017,2022))

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


def inter1d():
    import numpy as np
    from scipy import interpolate

    topic_yr_map_new = {}
    for i in pred_yrs:
        topic_yr_map_new[i] = list([])


    a = list(p_dict['topic_yr_vals'].values())

    for topic_number in range(10):
        x = p_dict['distinct_yrs']
        y = p_dict['topic_yr_vals'][topic_number]

        x_temp = list(x)
        y_temp = list(y)

        f = interpolate.interp1d(x_temp, y_temp, fill_value="extrapolate")

        for j in pred_yrs:
            topic_yr_map_new[j].append(max(0,float(f(j))))

    return topic_yr_map_new


# Main Fn Call
a = inter1d()
topic_yr_map.update(a)
print(x)
print(type(x))
print(pred_yrs)
print(type(pred_yrs))
x+=pred_yrs
print(x)

# print(topic_yr_map.keys())
viz_tot(topic_yr_map,x)
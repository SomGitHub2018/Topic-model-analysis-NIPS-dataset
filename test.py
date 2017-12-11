import numpy as np
import scipy
import scipy.interpolate
from scipy import *
import pickle
from scipy import interp, arange, exp

#load pickle
pickle_in = open("pickle_data.pickle","rb")
p_dict = pickle.load(pickle_in)
pickle_in.close()

topic_number = 5
x = p_dict['distinct_yrs']
y = p_dict['topic_yr_vals'][topic_number]

# print(interp([9,10,11,22], x, y))


# poly = np.polyfit(x[:-1], y[:-1], deg=3)
# print(np.polyval(poly, 10))

def sample():
    x_temp = list(x[:-1])
    y_temp = list(y[:-1])
    i = 2017

    tck = scipy.interpolate.splrep(x_temp, y_temp, k=1, s=0)
    new_val = scipy.interpolate.splev(i, tck)
    print(y[i], new_val)



def actual():
    x_temp = list(x[:-5])
    y_temp = list(y[:-5])
    for i in range(25,30):
        print(y_temp)

        tck = scipy.interpolate.splrep(x_temp, y_temp, k=1, s=0)
        new_val = scipy.interpolate.splev(i, tck)
        x_temp.append(i)
        y_temp.append(int(new_val))

        print(y[i], new_val)

def inter1d():
    import numpy as np
    from scipy import interpolate

    topic_number = 1
    x = p_dict['distinct_yrs']
    y = p_dict['topic_yr_vals'][topic_number]

    for i in range(10):
        x = p_dict['distinct_yrs']
        y = p_dict['topic_yr_vals'][topic_number]

        x_temp = list(x[:-3])
        y_temp = list(y[:-3])

        f = interpolate.interp1d(x_temp, y_temp, fill_value="extrapolate")
        print("topic_number ",topic_number)
        print(y[27],f(2014))
        print(y[28],f(2015))
        print(y[29],f(2016))
        print(" ")

        topic_number += 1


# sample()
# actual()
inter1d()
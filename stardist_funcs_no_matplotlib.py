import scipy.stats as st
#from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from glob import glob
from tqdm import tqdm
from tifffile import imread

from matplotlib import cm
from scipy import spatial

# Use this function to find the index of closest point in growth front 
# to the relevant osteoclast
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


growth_front_leniency = 60 #########What Quantile filter do you want remove


#load in cell coords with Kyles CSV


file_name = "/home/lpe/Downloads/results/190305_0_12hr_stats.csv"
def return_growth_front(file_name,growth_front_leniency):
    csv = np.genfromtxt (file_name, delimiter=",")
    first = csv[1:,0]
    second = csv[1:,1]
    time = csv[1:,2]
    distance_from_growth_front_master = []


    x = second
    y = first

    # Define the borders
    rxmin = min(x) #- deltaX
    rxmax = max(x) #+ deltaX
    rymin = min(y) #- deltaY
    rymax = max(y) #+ deltaY

    # I get nan and I dont know why..... so I a removing it, just being cautious
    # You prob have a better idea why theres an NaN in the time indicator col
    the_time_points = np.unique(time)
for time_point in the_time_points:

    dist_from_growth_front = []
    x = second[time == time_point]
    y = first[time == time_point]


    # Define the borders
    xmin = min(x) #- deltaX
    xmax = max(x) #+ deltaX
    ymin = min(y) #- deltaY
    ymax = max(y) #+ deltaY
    intxmax =  np.int(np.ceil(rxmax)) + 1
    intymax = np.int(np.ceil(rymax)) + 1
    mask = np.zeros((intymax, intxmax))

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:1000j, ymin:ymax:1000j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    import cv2
    ft = f.T
    ftint = (ft * 10000000000000).astype(int)
    resized = cv2.resize(ftint.astype(float), ((int(xmax) + 1),(int(ymax) + 1)), interpolation = cv2.INTER_NEAREST)
    quants = np.percentile(resized , growth_front_leniency)

    resized[resized < quants] = 0
    resized[resized != 0] = 255
    idx = cv2.findContours(resized.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[1][0]
    out = np.zeros_like(resized)
    out[idx[:,0,1],idx[:,0,0]] = 255
    make_the_mask = np.asarray([np.nonzero(out)[1],np.nonzero(out)[0]]).T


    for run, rise in zip(x,y):

        node = np.array([int(run) , int(rise)])
        nodes = make_the_mask
        in_or_out = resized[node[1],node[0]]

        closest_line =closest_node(node, nodes)
        closest_point_on_line = nodes[closest_line]
        dist = np.linalg.norm(closest_point_on_line - node)
        if in_or_out == 0:
            dist = dist * -1
        dist_from_growth_front.append(dist)
    distance_from_growth_front_master.append(dist_from_growth_front)
    
return(distance_from_growth_front_master,x,y,time)    
    #img_name = '/home/lpe/Downloads/results/'+ str(time_point)+'_' +'name.jpg'
    #cv2.imwrite(img_name, resized)

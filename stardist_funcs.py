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




#load in cell coords with Kyles CSV


file_name = "/home/lpe/Downloads/results/190305_0_12hr_stats.csv"
def return_growth_front(file_name):
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

    #plt.figure(figsize=(8,8))


    fig = plt.figure(figsize=(8,8))
    ax = fig.gca()
    ax.set_xlim(rxmin, rxmax)
    ax.set_ylim(rymin, rymax)






    # I get nan and I dont know why..... so I a removing it, just being cautious
    # You prob have a better idea why theres an NaN in the time indicator col
    the_time_points = np.unique(time)



    #Loop through the time points, at each time point find the growth front, use a gmm
    #to filtre out outliers -this a bit much IG but its convinient for me- 
    #Use the most lenient -largest- density kernel.

    # largest fragment of 2d lenient contour is the growth front

    #use contours to make the kernel into points then find the closest point of kernel to
    #the cell --- Now compute the cells disetance to grwoth fronto and upload it into new col

    growth_front = 1


    for time_point in the_time_points:
        print(time_point)

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
        cset = ax.contour(xx, yy, f)



        #print(growth_front)
        contour = len(cset.allsegs[growth_front])


        if contour > 1:
            counter = 0

            for fragment in cset.allsegs[growth_front]:
                counter =  counter + len(fragment)
                print(len(fragment))

            make_the_mask = np.zeros((counter,2))
            for fragment in cset.allsegs[growth_front]:
                choose = 0
                for element in fragment:

                    #print(element)
                    make_the_mask[choose ,:] = element

                    choose = choose + 1

            pts = np.array(make_the_mask, dtype=np.int32)
            #print((np.ceil(xmax), np.ceil(ymax)))

            intxmax =  np.int(np.ceil(rxmax)) + 1
            intymax = np.int(np.ceil(rymax)) + 1
            mask = np.zeros((intymax, intxmax))

            cv2.fillConvexPoly(mask, pts, 1)
            mask = mask.astype(np.bool)



            for run, rise in zip(x,y):

                node = np.array([int(run) , int(rise)])
                nodes = make_the_mask
                in_or_out = mask[node[1],node[0]]

                closest_line =closest_node(node, nodes)
                closest_point_on_line = nodes[closest_line]
                dist = np.linalg.norm(closest_point_on_line - node)
                if in_or_out == True:
                    dist = dist * -1
                dist_from_growth_front.append(dist)



        else :    
            #plt.plot(cset.allsegs[growth_front][0][:,0], cset.allsegs[growth_front][0][:,1])
        
        
        
        
            for fragment in cset.allsegs[growth_front]:
                counter =  counter + len(fragment)
                print(len(fragment))

            make_the_mask = np.zeros((counter,2))
            for fragment in cset.allsegs[growth_front]:
                choose = 0
                for element in fragment:

                    #print(element)
                    make_the_mask[choose ,:] = element

                    choose = choose + 1
            
            
            pts = np.array(make_the_mask, dtype=np.int32)
            pts = np.vstack((pts,[xmin,ymin]))
            pts = np.vstack((pts,[xmin,ymax]))
            pts = np.array(pts, dtype=np.int32)
            
            #print((np.ceil(xmax), np.ceil(ymax)))

            intxmax =  np.int(np.ceil(rxmax)) + 1
            intymax = np.int(np.ceil(rymax)) + 1
            mask = np.zeros((intymax, intxmax))

            cv2.fillConvexPoly(mask, pts, 1)
            mask = mask.astype(np.bool)

            for run, rise in zip(x,y):

                node = np.array([int(run) , int(rise)])
                nodes = cset.allsegs[growth_front][0]


                closest_line =closest_node(node, nodes)
                closest_point_on_line = nodes[closest_line]
                dist = np.linalg.norm(closest_point_on_line - node)
                in_or_out = mask[node[1],node[0]]

                if in_or_out == True:
                    dist = dist * -1
                dist_from_growth_front.append(dist)

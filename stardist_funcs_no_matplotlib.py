def contours_to_mask(path,ymax,xmax,ymin,xmin):
    
    full_path = np.array([0,2])
    
    segments = path.collections[1].get_paths()
    for segment in segments:
        verts = segment.vertices
        full_path =  np.vstack([full_path, verts])
    
    mask = np.zeros((int(ymax), int(xmax)))
    cv2.fillConvexPoly(mask, full_path.astype('int'), 1) 
    mask = mask.astype(np.bool)
    mask = mask[int(xmin):, int(ymin):]
    
    return(mask)

def difference_of_masks(earlier,later,ymax,xmax,ymin,xmin):

    early_mask = contours_to_mask(earlier_gf,rymax,rxmax,rymin,rxmin)
    later_mask = contours_to_mask(later_gf,rymax,rxmax,rymin,rxmin)

    diff_mask = ((later_mask * 1) - ( early_mask * 1))

    the_counts = sum((diff_mask.ravel() == 0 ) * 1)
    return(the_counts)

### in order to deal with the 'stupid contour  > 1 crap' you can buffer
### the meshgrid with with extra space so a full contour can be created
### with matplotlib....this makes everything downstream alot easier
### the code for this is on line 69
### the full contour is bounded by xmax and ymax during mask creation
### it is then bounded by ymin,xmin during mask creation
def growth_front_kde(file,buffer,frame):
    
    csv = np.genfromtxt (file_name, delimiter=",")
    first = csv[1:,0]
    second = csv[1:,1]
    time = csv[1:,2]

    x = second 
    y = first

    # Define the borders
    rxmin = min(x) #- deltaX
    rxmax = max(x) #+ deltaX
    rymin = min(y) #- deltaY
    rymax = max(y) #+ deltaY
    ax = fig.gca()

    ax.set_xlim((rxmin -buffer), (rxmax + buffer))#############
    ax.set_ylim((rymin - buffer), (rymax + buffer))############

    the_time_points = np.unique(time)

    growth_front = 1

    time_point = frame

    print(time_point)

    dist_from_growth_front = []
    x = second[time == time_point]
    y = first[time == time_point]
    intxmax =  np.int(np.ceil(rxmax)) + 1
    intymax = np.int(np.ceil(rymax)) + 1
    mask = np.zeros((intymax, intxmax))

    # Create meshgrid# IT MUST BE BUFFERED SO THAT A SINGLE CONTOUR IS GENERATED
    xx, yy = np.mgrid[(xmin - buffer):(xmax + buffer):2000j, (ymin -buffer):(ymax + buffer):2000j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    cset = ax.contour(xx, yy, f)
    return(cset)

# source: https://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.cluster import KMeans
from time import time
import pandas as pd
from sklearn import metrics
from sklearn.metrics.pairwise import manhattan_distances


# number of clusters
n_colors = 10   

image = plt.imread("2_animals_2.jpg")             

# convert to float and divide on 255 (it is needed for plot.imshow() function)
my_image_2d = np.array(image, dtype=np.float64) / 255

# let's transform 3d array of image to 2d array:
w, h, d = original_shape = tuple(my_image_2d.shape)
assert d == 3
# array 3d to 2d array
image_array = np.reshape(my_image_2d, (w * h, d))

# fit model
t0 = time() #let's see the time
kmeans = KMeans(n_clusters=n_colors).fit(image_array)
# k-means centers
cluster_centers = kmeans.cluster_centers_
# k-means classified pixels
labels = kmeans.labels_

print("done in %0.3fs." % (time() - t0))

# function that converts data calculated by k-means method 
# into data recognized by plot.imshow() function
def recreate_image(centers, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = centers.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = centers[labels[label_idx]]
            label_idx += 1
    return image

#function that gets points from cluster 
def getPoints(cluster):
    current_cluster = cluster_map[cluster_map.cluster == cluster]
    # get appropriate points wich belongs to this cluster
    points = np.asarray(current_cluster['coords'].tolist(), dtype=np.float16)
    # cut points (it is needed for manhattan_distances() function)
    cutted_points= points[:1000]
    return cutted_points

# let's calculate accuracy (chi beni criteria):

# create dataframe with two columns (cluster of point and coordinates of point)
cluster_map = pd.DataFrame({'cluster': labels, 'coords': list(image_array)}, columns=['cluster', 'coords'])

# inner distance in cluster
inner_cluster_distance = 0
# get unique labels (our clusters) from all labels
unique_labels = np.unique(labels)

# let's calculate inner cluster distance
# for any cluster
for cluster in unique_labels:
    cutted_points=getPoints(cluster)
    #calculate distances 
    md = manhattan_distances(cutted_points)
    # sum all distances
    sum_md= sum(sum(md))
    # append result to inner_cluster_distance
    inner_cluster_distance=inner_cluster_distance+sum_md

# let's calculate summery distance between differant clusters
# distance between diffarant clusters
distance_between_clusters = 0

for cluster1 in unique_labels:
    for cluster2 in unique_labels:
        # we need points from different clusters
        if cluster1!=cluster2:
            # cluster1
            cutted_pointns1=getPoints(cluster1)
            # cluster2
            cutted_pointns2=getPoints(cluster2)

            #calculate distances 
            md = manhattan_distances(cutted_pointns1,cutted_pointns2)
            # sum all distances
            sum_md= sum(sum(md))
            # append result to inner_cluster_distance
            distance_between_clusters=distance_between_clusters+sum_md
    
chi_beny=float(inner_cluster_distance/distance_between_clusters)
print("chi_beny="+str(chi_beny))

# Display results
# original image
# plt.figure(1)
# plt.clf()
# plt.axis('off')
# plt.title('Original image')
# plt.imshow(my_image_2d)

# clusterized image
plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image ('+ str(n_colors)+' colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))
plt.show()



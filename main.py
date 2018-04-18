import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import random
import copy
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

# Provides euclidian distance
def dist(a, b, axis=1):
    return np.linalg.norm(a - b, axis=axis)

def main():
    # Size of the K value
    k = 3
    
    # Load the csv file with pandas
    irisDF = pd.read_csv(sys.path[0] + r'\Iris.csv')
    
    # Split the data for later use.
    irisSetosaVals = irisDF.iloc[0:50, 1:5].values
    irisVersicolorVals = irisDF.iloc[50:100, 1:5].values
    irisVirginicaVals = irisDF.iloc[100:150, 1:5].values
    
    # PickirisSetosaValsntroids
    indexC1 = random.randint(0, 49)
    indexC2 = random.randint(0, 49)
    indexC3 = random.randint(0, 49)
    
    # Randomly select a point for each cluster.
    c1 = irisSetosaVals[indexC1]
    c2 = irisVersicolorVals[indexC2]
    c3 = irisVirginicaVals[indexC3]
    
    # Combine the clusters into one array
    c = np.vstack((c1, c2, c3))
    
    # The older clustr values.
    prevC = np.zeros((c.shape[0], c.shape[1]))
    err = np.sum(dist(prevC, c))

    npIris =  np.array(list(irisDF.iloc[0:150, 1:5].values))
    
    # This list will hold all of the clusters.
    clusters = [[] for i in range(k)]

    while(err != 0):
        # Reset the clusters.
        clusters = [[] for i in range(k)]

        # Assign the points 
        for i in range(npIris.shape[0]):
            dp = npIris[i]
            distances = dist(dp, c)
            minIndex = np.argmin(distances)
            clusters[minIndex].append(dp)
        
        # Store to compare error
        prevC = copy.deepcopy(c)
        
        # Average the clusters around values.
        for i in range(k):
            currentCluster = clusters[i]
            sumCluster = sum(currentCluster)
            avgCluster = sumCluster/len(currentCluster)
            c[i] = avgCluster

        err = np.sum(dist(c, prevC))
    
    # Combine the clusters into seprate variables
    cluster1 = np.vstack(clusters[0])
    cluster2 = np.vstack(clusters[1])
    cluster3 = np.vstack(clusters[2])

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()

    fig.canvas.set_window_title('SepalLengthCm and SepalWidthCm')
    fig2.canvas.set_window_title('PetalLengthCm and PetalWidthCm')
    
    # Plot the SepalLengthCm and SepalWidthCm
    ax.scatter(c[0:3, 0], c[0:3:,1] , marker='*' , c='black', s=200)
    ax.scatter(cluster1[:, 0], cluster1[:, 1], c='red', s=7)
    ax.scatter(cluster2[:, 0], cluster2[:, 1], c='blue', s=7)
    ax.scatter(cluster3[:, 0], cluster3[:, 1], c='green', s=7)
    
    # Plot the PetalLengthCm and PetalWidthCm
    ax2.scatter(c[0:3, 2], c[0:3:, 3] , marker='*' , c='black', s=200)
    ax2.scatter(cluster1[:, 2], cluster1[:, 3], c='red', s=7)
    ax2.scatter(cluster2[:, 2], cluster2[:, 3], c='blue', s=7)
    ax2.scatter(cluster3[:, 2], cluster3[:, 3], c='green', s=7)
    plt.show()
    

if __name__ == '__main__':
    main()
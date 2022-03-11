# Krzysztof Szymankiewicz, indeks 183216
# grupowanie danych przy wykorzystaniu implementacji algorytmu k-means

import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
from random import randrange

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def bestVariation(variation):
    avgVar = sum(variation) / len(variation)
    var = avgVar
    for el in variation:
        if el > avgVar:
            var -= el
        else:
            var += el
    return (avgVar - var)

def avgKclusters(k, x, y, size, clusterNum):
    avgX = [0] * k
    avgY = [0] * k
    count = [0] * k
    variation = [0] * k
    for i in range(0, size):
        count[clusterNum[i]] += 1
        avgX[clusterNum[i]] += x[i]
        avgY[clusterNum[i]] += y[i]
    for i in range(0, k):
        if count[i] != 0:
            avgX[i] /= count[i]
            avgY[i] /= count[i]

    for i in range(0, size):
        min = distance(x[i], y[i], avgX[0], avgY[0])
        distFromPoint = min
        for j in range(1, k):
            dist = distance(x[i], y[i], avgX[j], avgY[j])
            if dist < min:
                min = dist
                distFromPoint = dist
                clusterNum[i] = j
        variation[clusterNum[i]] += distFromPoint
    return clusterNum, variation


def kclusters(k, x, y, size):
    numOfTries = 10
    randomPointsX = [0] * k
    randomPointsY = [0] * k
    startingPointsX = [[0 for a in range(numOfTries)] for b in range(k)]
    startingPointsY = [[0 for a in range(numOfTries)] for b in range(k)]
    variations = [0] * numOfTries
    savedClusters = [[0 for a in range(size)] for b in range(numOfTries)]
    for l in range(0, numOfTries):
        clusterNum = [0] * size
        for i in range(0, k):
            randPos = randrange(size)
            randomPointsX[i] = x[randPos]
            randomPointsY[i] = y[randPos]
            startingPointsX[i][l] = randomPointsX[i]
            startingPointsY[i][l] = randomPointsY[i]
        for i in range(0, size):
            minDist = distance(x[i], y[i], randomPointsX[0], randomPointsY[0])
            distFromPoint = minDist
            for j in range(1, k):
                dist = distance(x[i], y[i], randomPointsX[j], randomPointsY[j])
                if dist < minDist:
                    minDist = dist
                    distFromPoint = dist
                    clusterNum[i] = j
            savedClusters[l] = clusterNum
        for i in range(0, 10):
            savedClusters[l], variation = avgKclusters(k, x, y, size, savedClusters[l])
        variations[l] = variation

    bestVariation = min(variations, key=sum)
    bestVariationPos = variations.index(bestVariation)
    clusterNum = [0] * size
    variation = [0] * size

    return savedClusters[bestVariationPos], bestVariation

def unknownkclusters(k, x, y, size):
    maxK = 10
    variationK = [0] * (maxK - 2)
    savedClusterNum = [0] * (maxK - 2)
    if k > 0:
        savedClusterNum[k - 2], variation = kclusters(k, x, y, size)
        variationK[k - 2] = bestVariation(variation)
    else:
        for i in range(2, maxK):
            savedClusterNum[i - 2], variation = kclusters(i, x, y, size)
            variationK[i - 2] = bestVariation(variation)
    minVariation = variationK.index(min(i for i in variationK if i != 0))
    finalK = minVariation + 2
    clusterNum = savedClusterNum[minVariation]
    print(finalK)
    return finalK, clusterNum

def compareToOriginalGroups(k, clusters, oriClusters, size):
    sum = [[0 for a in range(k)] for b in range(k)]
    ordered = [0] * k
    difference = [0] * size
    for i in range(k):
        for j in range(k):
            for l in range(size):
                if clusters[l] == i and oriClusters[l] - 1 == j:
                    sum[i][j] += 1
    for i in range(k):
        ordered[i] = sum[i].index(max(sum[i]))
    print(ordered)
    for i in range(size):
        if ordered[clusters[i]] != oriClusters[i] - 1:
            difference[i] = 1
    return difference


data = pd.read_csv('data/seeds_dataset.csv')
print(data)
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
colorsError = ['black', 'red']

x = data.Perimeter
y = data.Compactness

k, clusterNum = unknownkclusters(0, x, y, data.index.stop)
difference = compareToOriginalGroups(k, clusterNum, data.Group, data.index.stop)

figure, axes = plt.subplots()
plt.xlabel("Perimeter")
plt.ylabel("Compactness")
for i in range(0, data.index.stop):
    plt.scatter(x[i], y[i], color=colors[clusterNum[i]])
plt.title("k = " + str(k))
plt.savefig("kmeans_plots/Perimeter_Compactness" + str(k) + ".png")
'''
for l in range(data.index.stop):
    plt.scatter(x[l], y[l], color=colorsError[difference[l]])
plt.title("error = " + str(sum(difference)))
plt.show()
'''
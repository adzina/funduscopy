import random
import numpy as np
import math
import operator
import cv2
import testData


def randPixels(array, pixelNumber):
    """Picks pixelNumber items from 2d array. Returns list of coords (pairs)."""
    x = len(array)
    y = len(array[0])
    coords = []

    for _ in range(pixelNumber):
        pick = [random.randrange(x), random.randrange(y)]
        while pick in coords:
            pick = [random.randrange(x), random.randrange(y)]
        coords.append(pick)

    return coords


def isThatFundus(coords, array):
    """Returns true if array item is a fundus/white pixel/not [0 0 0] numpy array"""
    if np.array_equal(array[coords[0]][coords[1]], np.zeros(3)):
        return False
    return True


def fundusDatabase(array, pixelNumber):
    """Returns db with "fundus" list and "else" list"""
    coords = randPixels(array, pixelNumber)
    db = {}

    for c in coords:
        lst = []
        if isThatFundus(c, array):
            if "fundus" in db.keys(): lst = db["fundus"]
            lst.append(c)
            db.update({"fundus": lst})
        else:
            if "else" in db.keys(): lst = db["else"]
            lst.append(c)
            db.update({"else": lst})

    return db


def countHuMoments(array):
    array = np.array(array)
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    return cv2.HuMoments(cv2.moments(array)).flatten()


def countVariance(array):
    return np.var(array)

def countAverage(array):
    '''returns mean value for r g b in form [mean_r mean_g mean_b]'''
    return array.mean(axis = 0)

def cut25x25FromArray(array, coords):
    """Returns 25x25 square from given array (or less when start point is close to the edge)"""
    result = []
    pom = []

    for x in range(coords[0]-12, coords[0]+13):
        for y in range(coords[1]-12, coords[1]+13):
            if x >= 0 and y >= 0 and x < len(array) and y < len(array[0]):
                pom.append(array[x][y])
        if pom != []: result.append(pom)
        pom = []

    return result


def countPixelParameters(baseArray, coords):
    '''Returns an array of parameters for a given pixel
        [average, Hu moment, colour variance]
    '''
    baseArray = cut25x25FromArray(baseArray, coords)
    average = countAverage(np.array(baseArray))
    #huMoments = countHuMoments(baseArray)
    huMoments = [0,0,0,0,0,0,0]
    colourVar = countVariance(baseArray)
    return [average, huMoments, colourVar]


def countAllParameters(baseArray):
    '''
     Every pixel is converted to an array of parameters
    '''
    paramArray = []
    for x in range(len(baseArray)):
        row=[]
        for y in range(len(baseArray[0])):
            c=[x,y]
            row.append(countPixelParameters(baseArray, c))
        paramArray.append(row)
    return paramArray


def euclideanDistance(pixel1, pixel2):
    '''
    Returns euclidian distance between two pixels.
    '''
    distance = 0
    # distance between averages
    distance += pow((pixel1[0] - pixel2[0]),2)
    # sum of distances of every hu value
    distance += pow(euclidianDistanceHu(pixel1[1],pixel2[1]),2)
    # distance between variances
    distance += pow((pixel1[2] - pixel2[2]), 2)

    return math.sqrt(distance)

def euclidianDistanceHu(hu1, hu2):
    '''
        Returns euclidian distance between twoarrays of hu moments.
        '''
    distance = 0
    for i in range(7):
        distance += pow((hu1[i]-hu2[i]),2)
    return math.sqrt(distance)

def getNeighbours(trainingSet, pixel, k):
    '''
    Returns k closest neighbours of a pixel.
    '''
    distances = []
    length = len(pixel)-1
    for x in range(len(trainingSet)):
        '''ommit first elem of trainingSet rows: it says if pixel is a fundus'''
        dist = euclideanDistance(pixel, trainingSet[x][1:])
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbours = []
    for x in range(k):
        neighbours.append(distances[x][0])
    return neighbours


def predictFundus(coords, paramArray):
    trainingSet = testData.readTrainingSet()
    fundus = 0
    nonFundus = 0
    k=4
    neighbours = getNeighbours(trainingSet, paramArray[coords[0],coords[1]],k)
    for i in neighbours:
        if i[0]==True: #isFundus
            fundus+=1
        else:
            nonFundus+=1
    return fundus>nonFundus


def generateBinaryImage(image):
    paramArray = countAllParameters(image)
    image=np.array(image)
    binary=np.zeros(image.shape)
    for x in range(len(image)):
        for y in range(len(image[0])):
            c=[x,y]
            if predictFundus(c,paramArray):
                binary[x][y]=[255,255,255]
            else:
                binary[x][y]=[0,0,0]

    return binary

if __name__ == "__main__":
    # main function for tests
    test = []
    pom = []
    for x in range(12):
        for y in range(3):
            pom.append(x)
        test.append(pom)
        pom = []
    print(cut25x25FromArray(test, [11, 11]))
    cut = cut25x25FromArray(test, [11, 11])

    print(cut)
    print(countAllParameters(np.array(cut)))


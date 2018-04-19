import random
import numpy as np
import math
import operator

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


def countFundusNeighbours(array, coords):
    allNeighbours = 0
    fundusNeighbours = 0

    for x in range(coords[0]-4, coords[0]+5):
        for y in range(coords[1]-4, coords[1]+5):
            if x >= 0 and y >= 0 and x < len(array) and y < len(array[0]):
                if x == coords[0] and y == coords[1]:
                    continue  # skip start coords
                allNeighbours += 1
                if isThatFundus([x, y], array):
                    fundusNeighbours += 1

    return fundusNeighbours, allNeighbours

def countPixelParameters(array, coords):
	'''Returns an array of parameters for a given pixel
		[fundusNeighbours, Hu moment, colour variance]
	'''
	fundusNeighbours = countFundusNeighbours(array,coords)
	fundusNeighbours = 0
	# huMoment = countHuMoment(array,coords)
	huMoment = 0
	# colourVar = countColourVar(arry,coords)
	colourVar = 0
	
	return [fundusNeighbours, huMoment, colourVar]

def countAllParameters(array):
	'''
	 Every pixel is converted to an array of parameters
	'''
	paramArray = []
	for x in range(len(array)):
		row=[]
		for y in range(len(array[0])):
			c=[x,y]
			row.append(countPixelParameters(array,c))
		paramArray.append(row)
	return paramArray	
	
def euclideanDistance(pixel1, pixel2, length):
	'''
	Returns euclidian distance between two pixels.
	Length is the number of properties each pixel has.
	'''
	distance = 0
	for x in range(length):
		distance += pow((pixel1[x] - pixel2[x]), 2)
	return math.sqrt(distance)	
	
def getNeighbours(trainingSet, pixel, k):
	'''
	Returns k closest neighbours of a pixel.
	'''
	distances = []
	length = len(pixel)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(pixel, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbours = []
	for x in range(k):
		neighbours.append(distances[x][0])
	return neighbours	
def generateBinaryImage(image):
	image=np.array(image)
	binary=np.zeros(image.shape)
	for x in range(len(image)):
		row=[]
		for y in range(len(image[0])):
			c=[x,y]
			if(isThatFundus(c,image)):
				binary[x][y]=[255,255,255]
			else:
				binary[x][y]=[0,0,0]
		
	return binary			
if __name__ == "__main__":
    # main function for tests
    test = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
    a = [2,3,4,5,6]
    neighbours = getNeighbours(test, a,2)
    print(neighbours)
    

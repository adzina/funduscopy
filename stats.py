import random
import numpy as np


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


if __name__ == "__main__":
    # main function for tests
    test = [[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
    # a,b = countFundusNeighbours(test, [0,0])
    # print(a)
    # print(b)

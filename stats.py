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


def whatIsThat(coords, array):
    """Returns true if array item is a fundus/white pixel/[0 0 0] numpy array"""
    if np.array_equal(array[coords[0]][coords[1]], np.zeros(3)):
        return True
    return False


#  nana
def fundusDatabase(array, pixelNumber):
    """Returns db with "fundus" list and "else" list"""
    coords = randPixels(array, pixelNumber)
    db = {}

    for c in coords:
        lst = []
        if whatIsThat(c, array):
            if "fundus" in db.keys(): lst = db["fundus"]
            lst.append(c)
            db.update({"funds": lst})
        else:
            if "else" in db.keys(): lst = db["else"]
            lst.append(c)
            db.update({"else": lst})

    return db

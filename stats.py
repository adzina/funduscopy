import random
import numpy as np
import math
import operator
import cv2
import testData
import matplotlib as plt
from sklearn.metrics import mean_squared_error
from PIL import Image

#coefficiants of parameters' significance: [average, hu, variance]
global COEF
global DILATATION
COEF = [2, 100000, 1]
DILATATION = 10

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
    r = []
    g = []
    b = []
    for i in array:
        r.append(i[0])
        g.append(i[1])
        b.append(i[2])
    mean_r = np.array(r).mean()		
    mean_g = np.array(g).mean()	
    mean_b = np.array(b).mean()
    return np.array([mean_r, mean_g, mean_b])

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
    baseArrayCut = cut25x25FromArray(baseArray, coords)
	#substract pixel colour from average of neighbours' colour. If value close to 0 than probably pixel is not a fundus
    
    average = baseArray[coords[0]][coords[1]] - countAverage(np.array(baseArrayCut))
    	
    huMoments = countHuMoments(baseArrayCut)
    colourVar = countVariance(baseArrayCut)
    average, huMoments, colourVar = scaleParameters(average, huMoments, colourVar)
    		
    return [average, huMoments, colourVar]

def scaleParameters(average, huMoments, colourVar):
    '''Scales the parameters so that all contain values between roughly 0 and 1'''
    colourVar = colourVar/10000
    average = average/255
    hu_min_max= [[1.3e-03,2.7e-03],[8.0e-13,9.0e-08],[2.0e-15,2.1e-07],[1.0e-14, 2.0e-11],[-2.4e-28,1.5e-22],[-9.0e-19, 5.8e-16],[-3.5e-23, 4.5e-23]]
    
    for i in range(len(huMoments)):
        hu_min = hu_min_max[i][0]
        hu_max = hu_min_max[i][1]
        scope = (hu_max - hu_min)
        part = huMoments[i] - hu_min
        
        huMoments[i] = part/scope
	
    return (average, huMoments, colourVar)

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
    distance += (pow((pixel1[0][0] - pixel2[0][0]),2)+ pow((pixel1[0][1] - pixel2[0][1]),2) + pow((pixel1[0][2] - pixel2[0][2]),2))* COEF[0]
    # sum of distances of every hu value
    distance += pow(euclidianDistanceHu(pixel1[1],pixel2[1]),2) * COEF[1]
    # distance between variances
    distance += pow((pixel1[2] - pixel2[2]), 2) * COEF[2]
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


def contoursApprox(image):
    """
    Takes image and returns filtered contours.
    Return type: 2D np array with values 0 (not fundus) or 255 (fundus).
    """

    def setImage(image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    def setBlur(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # conversion, getting basic pic

        image = cv2.GaussianBlur(image, (3, 3),  # gaussian kernel size
                                 0)  # sigmaX - gaussian kernel standard derivation in x direction

        image = cv2.medianBlur(image, 3)  # aperture linear size

        return image

    def getThresh(image):
        t, _ = cv2.threshold(image, 0,  # threshold value
                             255,  # max value to use with thresh_bin
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresh type
        return t

    def applyFilters(image, operations):
        # canny - edge detection
        if "canny" in operations.keys():
            image = cv2.Canny(image, 0.2 * operations["canny"],  # first threshold
                              operations["canny"])  # 2nd thresh for hysteresis

        # dilatation - increases thickness
        if "dilatation" in operations.keys():
            image = cv2.dilate(image, np.ones((3, 3), np.uint8),  # dilatation structuring element
                               iterations=operations["dilatation"])  # repeat number

        # erosion - opposite to dilatation
        if "erosion" in operations.keys():
            image = cv2.erode(image, np.ones((3, 3), np.uint8),  # erosion structuring element
                              iterations=operations["erosion"])  # repeat number

        return image

    def getContours(image):
        _, c, _ = cv2.findContours(image, cv2.RETR_EXTERNAL,  # retrieval mode, only extreme outer contours
                                   cv2.CHAIN_APPROX_SIMPLE)  # approx method
        return c
    

    image = setImage(image)

    # drawing first contour
    blurred = setBlur(image)

    # thresh - changing value if bigger to one, if lower to another
    thresh = getThresh(blurred)

    filtered = applyFilters(blurred, {"canny": thresh, "dilatation": DILATATION})

    # plt.imshow(filtered)

    return filtered

def calcDiffByArray(calculatedResult, pathFile):
    """
    calculatedResult - black'n'white matrix, computation result
    pathFile - full path of base file, used to pick right result from result_img
    """
    pathFile = pathFile.replace(".ppm", ".ah.ppm")
    pathFile = pathFile.replace("test_img", "result_img")
    im = Image.open(pathFile)
    # im.show()
    INPUT = list(im.getdata())
    RESULT = list(calculatedResult)  # TODO is type ok?

    return mean_squared_error(INPUT, RESULT)


def countStats():
        img_col = cv2.imread('test_img/im0001.ppm')
        img_bw = cv2.imread('result_img/im0001.ah.ppm')
        contours = contoursApprox(img_col)
        false_positive = 0
        false_negative = 0
        for i in range(len(contours)):
            for j in range(len(contours[0])):
                if( np.array_equal(contours[i][j],0) and  np.array_equal(img_bw[i][j],[255,255,255])):
                    false_positive+=1
                if( np.array_equal(contours[i][j],255) and  np.array_equal(img_bw[i][j],[0,0,0])):
                    false_negative+=1
        file = open("dilatation_stats.txt","a")
        str_to_write = "DILATATION: "+str(DILATATION)+ \
        "\n"+"false_positive: "+str(round(100*false_positive/(len(contours)*len(contours[0])),2))+"%\n"+ \
        "false negative: "+str(round(100*false_negative/(len(contours)*len(contours[0])),2))+"%\n\n"
        file.write(str_to_write)
        file.close()
def predictFundus(coords, paramArray, trainingSet):
    fundus = 0
    nonFundus = 0
    k=3
    neighbours = getNeighbours(trainingSet, paramArray[coords[0]][coords[1]],k)
    for i in neighbours:
        if i[0]==True: #isFundus
            fundus+=1
        else:
            nonFundus+=1
    return fundus>nonFundus


def generateBinaryImage(image):
    paramArray = countAllParameters(image)
    trainingSet = testData.readTrainingSet()[:1000]
    print("read TrainingSet")
    binary = []
    for x in range(len(image)):
        row = []
        for y in range(len(image[0])):
            c=[x,y]
            if (not predictFundus(c,paramArray, trainingSet)) or x==0 or y==0 or x==len(image)-1 or y==len(image[0])-1 :
                row.append(0)
            else:
                row.append(255)
        binary.append(row)
        print("row "+str(x)+" calculated")

    return binary

def countKNNStats():
    files = ["0002","0003","0004","0005","0044"]
    coefs = [[1, 10, 1],[1, 100, 1],[1, 100, 2],[1, 1000, 1],[1, 10000, 1],[1,100000, 1],[2,100000, 1]]
    for x in files:
        img_col = np.array(cv2.imread('test_img/im'+x+'.ppm'))
        img_bw = np.array(cv2.imread('result_img/im'+x+'.ah.ppm'))
        off_x = 200
        off_y = off_x
        size_x = 100
        size_y = size_x
        a = img_col[off_y:off_y+size_y]
        cut=[]
        for i in a:
            cut.append(i[off_x:off_x+size_x])
        cut = np.array(cut).flatten().reshape((size_x,size_y,3))
        
        for coef in coefs:
            COEF = coef
           
            contours = generateBinaryImage(cut)
            false_positive = 0
            false_negative = 0
            true_positive = 0
            true_negative = 0
            for i in range(len(contours)):
                for j in range(len(contours[0])):
                    if( np.array_equal(contours[i][j],0) and  np.array_equal(img_bw[i][j],[255,255,255])):
                        false_negative+=1
                    if( np.array_equal(contours[i][j],255) and  np.array_equal(img_bw[i][j],[0,0,0])):
                        false_positive+=1
                    if( np.array_equal(contours[i][j],255) and  np.array_equal(img_bw[i][j],[255,255,255])):
                        true_positive+=1
                    if( np.array_equal(contours[i][j],0) and  np.array_equal(img_bw[i][j],[0,0,0])):
                        true_negative+=1
            file = 'knn_stats/knn_stats_'+x+'.txt'
            file = open(file,"a")
            print(true_positive)
            print(true_negative)
            print(false_positive)
            print(false_negative)
            print(true_positive/(true_positive+true_negative))
            print(true_positive/(true_positive+false_positive))
            false_positive = round(false_positive,5)
            false_negative = round(false_negative,5)
            recall = round(true_positive/(true_positive+true_negative),5)
            precision = round(true_positive/(true_positive+false_positive),5)


            mse = calcDiffByArray(contours, "E:/studia/Informatyka/semestr VI/Informatyka w Medycynie/funduscopy/test_img/im"+x+'.ppm')
            str_to_write = "COEF: "+str(COEF)+ \
            "\n"+"false positive: "+str(false_positive)+"\n"+ \
            "false negative: "+str(false_negative)+"\n"+ \
			"recall: "+str(recall)+"\n"+ \
            "precision: "+str(precision)+"\n"+\
            "mse:"+str(mse)+"\n\n"
            file.write(str_to_write)
            file.close()

if __name__ == "__main__":
    # main function for tests
	countKNNStats()


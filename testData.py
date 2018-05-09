import stats
import cv2
from os import listdir

def readTrainingSet():
    files = listdir("test_results")
    files = [x[:-4] for x in files]
    trainingSet = []
    for file in files:
        with open("/home/odys1528/PycharmProjects/funduscopy/test_results/" + file + ".txt",
                  "r") as file:
            data = file.read()
            data = data[:-1]  # deleting last ] char
            tokens = data.split(",")
            i = 0
            while (i < len(tokens)):
                if(tokens[i].strip()[1:]=="False"):
                    isFundus = False
                else:
                    isFundus = True
                i += 1
                average=[]
                average.append(float(tokens[i][7:]))
                i += 1
                average.append(float(tokens[i]))
                i += 1
                average.append(float(tokens[i][:-2]))
                i += 1
                hu = []
                hu.append(float(tokens[i][8:].strip()))
                i += 1
                hu.append(float(tokens[i].strip()))
                i += 1
                hu.append(float(tokens[i].strip()))
                i += 1
                hu.append(float(tokens[i].strip()))
                i += 1
                hu.append(float(tokens[i].strip()))
                i += 1
                hu.append(float(tokens[i].strip()))
                i += 1
                hu.append(float(tokens[i][:-2].strip()))
                i += 1
                variance = float(tokens[i][:-1].strip())
                i += 1
                trainingSet.append((isFundus, average, hu, variance))
    return trainingSet				
if __name__ == "__main__":

    files = listdir("test_img")
    files = [x[:-4] for x in files]
    print(files)

    for file in files:
       resultImg = cv2.imread("E:/studia/Informatyka/semestr VI/Informatyka w Medycynie/funduscopy/result_img/" + file + ".ah.ppm")  # b&w
       baseImg = cv2.imread("E:/studia/Informatyka/semestr VI/Informatyka w Medycynie/funduscopy/test_img/" + file + ".ppm")  # color
       print("file: " + file)

       file = open("E:/studia/Informatyka/semestr VI/Informatyka w Medycynie/funduscopy/test_results/" + file + ".txt", "w")

       db = stats.fundusDatabase(resultImg, 10000)
       i=0
       while len(db['fundus'])<300:
           i+=100
           db = stats.fundusDatabase(resultImg, 10000+i)
           print('again ' + str(len(db['fundus'])))

       print("db done")

       file.write("[")
       testPixels = db['fundus'][:300]+db['else'][:700]
       sep=", "
       for i in range(len(testPixels)):
           if testPixels[i] == testPixels[-1]: sep=""
           if(i<300):
               file.write("[True,"+str(stats.countPixelParameters(baseImg, testPixels[i]))[1:] + sep)
           else:
               file.write("[False," +str(stats.countPixelParameters(baseImg, testPixels[i]))[1:] + sep)
                 			
       file.write("]")
       file.close()

       print("file written")
       print("\n")

import stats
import cv2
from os import listdir

def readTrainingSet():
    files = listdir("test_results")
    files = [x[:-4] for x in files]
    trainingSet = []
    for file in files:
        with open("E:/studia/Informatyka/semestr VI/Informatyka w Medycynie/funduscopy/test_results/" + file + ".txt",
                  "r") as file:
            data = file.read()
            data = data[:-1]  # deleting last ] char
            tokens = data.split(",")
            i = 0
            while (i < len(tokens)):
                isFundus = bool(tokens[i][2:])
                i += 1
                average=[]
                average.append(float(tokens[i][8:]))
                i += 1
                average.append(float(tokens[i]))
                i += 1
                average.append(float(tokens[i][:-2]))
                i += 1
                hu = []
                hu.append(float(tokens[i][9:].strip()))
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
                print(trainingSet)
if __name__ == "__main__":
    files = listdir("test_img")
    files = [x[:-4] for x in files]
    print(files)

    for file in files:
        resultImg = cv2.imread("/home/odys1528/PycharmProjects/funduscopy/result_img/" + file + ".ah.ppm")  # b&w
        baseImg = cv2.imread("/home/odys1528/PycharmProjects/funduscopy/test_img/" + file + ".ppm")  # color
        print("file: " + file)

        file = open("/home/odys1528/PycharmProjects/funduscopy/test_results/" + file + ".txt", "w")

        db = stats.fundusDatabase(resultImg, 10000)
        i=0
        while len(db['fundus'])<500:
            i+=100
            db = stats.fundusDatabase(resultImg, 10000+i)
            print('again ' + str(len(db['fundus'])))

        # print(str(len(db['fundus'])) + " " + str(len(db['else'])))
        print("db done")

        file.write("[")
        testPixels = db['fundus'][:500]+db['else'][:500]
        sep=", "
        for c in testPixels:
            if c == testPixels[-1]: sep=""
            file.write(str(stats.countPixelParameters(baseImg, resultImg, c)) + sep)
        file.write("]")
        file.close()

        print("file written")
        print("\n")

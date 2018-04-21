import stats
import cv2
from os import listdir


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
        sep=","
        for c in testPixels:
            if c == testPixels[-1]: sep=""
            file.write(str(stats.countPixelParameters(baseImg, resultImg, c)) + sep)
        file.write("]")
        file.close()

        print("file written")
        print("\n")

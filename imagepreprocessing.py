import cv2
import numpy as np
import math
import os


def panelAbstract(srcImage):
    #   read pic shape
    imgHeight, imgWidth = srcImage.shape[:2]
    imgHeight = int(imgHeight)
    imgWidth = int(imgWidth)
    # 二維轉一維
    imgVec = np.float32(srcImage.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, clusCenter = cv2.kmeans(imgVec, 2, None, criteria, 10, flags)
    clusCenter = np.uint8(clusCenter)
    clusResult = clusCenter[label.flatten()]
    imgres = clusResult.reshape(srcImage.shape)
    # image轉成灰階
    imgres = cv2.cvtColor(imgres, cv2.COLOR_BGR2GRAY)
    # image轉成2維，並做Threshold
    _, thresh = cv2.threshold(imgres, 127, 255, cv2.THRESH_BINARY_INV)

    threshRotate = cv2.merge([thresh, thresh, thresh])
    # 印出 threshold後的image
    # if cv2.imwrite(r"./Photo/thresh.jpg", threshRotate):
    #    print("Write Images Successfully")
    # 确定前景外接矩形
    # find contours
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minvalx = np.max([imgHeight, imgWidth])
    maxvalx = 0
    minvaly = np.max([imgHeight, imgWidth])
    maxvaly = 0
    maxconArea = 0
    maxAreaPos = -1
    for i in range(len(contours)):
        if maxconArea < cv2.contourArea(contours[i]):
            maxconArea = cv2.contourArea(contours[i])
            maxAreaPos = i
    objCont = contours[maxAreaPos]

    # cv2.minAreaRect生成最小外接矩形
    rect = cv2.minAreaRect(objCont)
    for j in range(len(objCont)):
        minvaly = np.min([minvaly, objCont[j][0][0]])
        maxvaly = np.max([maxvaly, objCont[j][0][0]])
        minvalx = np.min([minvalx, objCont[j][0][1]])
        maxvalx = np.max([maxvalx, objCont[j][0][1]])
    if rect[2] <= -45:
        rotAgl = 90 + rect[2]
    else:
        # 咖啡粉會執行else
        rotAgl = rect[2]
    if rotAgl == 0:
        panelImg = srcImage[minvalx:maxvalx, minvaly:maxvaly, :]
    else:
        # 咖啡粉會執行else
        rotCtr = rect[0]
        rotCtr = (int(rotCtr[0]), int(rotCtr[1]))
        rotMdl = cv2.getRotationMatrix2D(rotCtr, rotAgl, 1)
        imgHeight, imgWidth = srcImage.shape[:2]
        # 圖像旋轉
        dstHeight = math.sqrt(imgWidth * imgWidth + imgHeight * imgHeight)
        dstRotimg = cv2.warpAffine(threshRotate, rotMdl, (int(dstHeight), int(dstHeight)))
        dstImage = cv2.warpAffine(srcImage, rotMdl, (int(dstHeight), int(dstHeight)))
        dstRotimg = cv2.cvtColor(dstRotimg, cv2.COLOR_BGR2GRAY)
        _, dstRotBW = cv2.threshold(thresh, 127, 255, 0)
        # 印出最小外接矩形
        # if cv2.imwrite(r"./Photo/squre.jpg", dstRotBW):
        #        print("Write Images Successfully")
        _, contours, hierarchy = cv2.findContours(dstRotBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0;
        maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x, y, w, h = cv2.boundingRect(contours[maxAreaPos])
        print(x, y, w, h)
        w = w / 8;  # 寬度分為8等分
        h = h / 4;  # 高度分為4等分
        # 印出切齊圓形的正方形
        # testImg = dstImage[int(y):int(y+4*h),int(x):int(x+8*w),:]
        # if cv2.imwrite(r"./Photo/dst.jpg", testImg):
        #        print("Write Images Successfully")

        # 將沒有外圍輪廓的咖啡粉存入panelImg
        panelImg = dstImage[int(y + h):int(y + 3 * h), int(x + w):int(x + 7 * w), :]

    return panelImg


if __name__ == "__main__":

    data_path = r"/home/ecl-123/zing/coffee_sever"

    for root_Outer, dirs_Outer, files_Outer in os.walk(data_path, topdown=False):
        for directory in dirs_Outer:
            for root, dirs, files in os.walk(os.path.join(root_Outer, directory), topdown=False):
                for name in files:
                    img_path = os.path.join(root, name)
                    srcImage = cv2.imread(img_path)
                    rstImage = panelAbstract(srcImage)
                    # 印出結果
                    filename = '%s_result.%s' % (name.split('.')[0], name.split('.')[-1])
                    print('new_Filename: ' + filename)
                    if cv2.imwrite(os.path.join(root, filename), rstImage):
                        print("Write Images Successfully")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

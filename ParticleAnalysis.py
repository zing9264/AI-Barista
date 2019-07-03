import sys
import math
import cv2
import numpy as np
import random as rng
from time import time
from imagepreprocessing import panelAbstract

show_Image = False


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
    # if # cv2.imwrite(r"./Photo/thresh.jpg", threshRotate):
    #    print("Write Images Successfully")
    # 确定前景外接矩形
    # find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        # if # cv2.imwrite(r"./Photo/squre.jpg", dstRotBW):
        #        print("Write Images Successfully")
        contours, hierarchy = cv2.findContours(dstRotBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0
        maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x, y, w, h = cv2.boundingRect(contours[maxAreaPos])
        print(x, y, w, h)

        umsize = 90000/w

        w = w / 8  # 寬度分為8等分
        h = h / 4  # 高度分為4等分
        # 印出切齊圓形的正方形
        # testImg = dstImage[int(y):int(y+4*h),int(x):int(x+8*w),:]
        # if # cv2.imwrite(r"./Photo/dst.jpg", testImg):
        #        print("Write Images Successfully")

        # 將沒有外圍輪廓的咖啡粉存入panelImg
        panelImg = dstImage[int(y + h):int(y + 3 * h), int(x + w):int(x + 7 * w), :]
        print("Image Size:", 2 * h * umsize, " um * ", 6 * w * umsize, " um")
        real_height = 2 * h * umsize
        real_width = 6 * w * umsize
        return panelImg, real_height, real_width

    return panelImg, 0, 0


def hist_equal_color(img):
    global show_Image

    # Converting image to LAB Color model
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Splitting the LAB image to different channels
    l, a, b = cv2.split(lab)
    if show_Image:
        cv2.namedWindow('l_channel', cv2.WINDOW_NORMAL)
        cv2.imshow('l_channel', l)
        cv2.namedWindow("a_channel", cv2.WINDOW_NORMAL)
        cv2.imshow('a_channel', a)
        cv2.namedWindow("b_channel", cv2.WINDOW_NORMAL)
        cv2.imshow('b_channel', b)

    # Applying CLAHE to L-channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to RGB model
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if show_Image:
        cv2.namedWindow("Hist Equal Color Result", cv2.WINDOW_NORMAL)
        cv2.imshow('Hist Equal Color Result', final)

    # cv2.imwrite("Hist-Equal-Color-Result.jpg",final)

    return final


def hist_equal_color_old(img):
    global show_Image
    # convert BGR image to YCbCr
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    # Equalizes the histogram of Y(Luma) channel in YCbCr
    channels = cv2.split(ycrcb)
    print(len(channels))
    if show_Image:
        cv2.namedWindow("Y channel", cv2.WINDOW_NORMAL)
        cv2.imshow("Y channel", channels[0])
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)

    # convert image back to BGR and return it
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def export_final_image(contours, markers):
    global show_Image
    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    print(markers.shape[0], markers.shape[1])

    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    # Visualize the final image
    if show_Image:
        cv2.namedWindow('Final Result', cv2.WINDOW_NORMAL)
        cv2.imshow('Final Result', dst)

    # Write final image to disk
    if cv2.imwrite(r"./Photo/Result.jpg", dst):
        print("Write Image Successfully")


def show_marker(markers):
    global show_Image
    mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)

    if show_Image:
        cv2.namedWindow('Markers_v2', cv2.WINDOW_NORMAL)
        cv2.imshow('Markers_v2', mark)

    # cv2.imwrite("Markers_v2.jpg", mark)



def gen_marker(dist, contours):
    global show_Image
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv2.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker
    cv2.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    if show_Image:
        cv2.namedWindow('Markers', cv2.WINDOW_NORMAL)
        cv2.imshow('Markers', markers)
    # cv2.imwrite("Markers.jpg", markers)

    return markers


def get_distribution(img, pixel2um):
    global show_Image

    rng.seed(12345)

    src = hist_equal_color(img)

    ## [sharp]
    # Create a kernel that we will use to sharpen our image
    # an approximation of second derivative, a quite strong kernel
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    # do the laplacian filtering as it is
    # well, we need to convert everything in something more deeper then CV_8U
    # because the kernel has some negative values,
    # and we can expect in general to have a Laplacian image with negative values
    # BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
    # so the possible negative number will be truncated
    imgLaplacian = cv2.filter2D(src, cv2.CV_32F, kernel)
    sharp = np.float32(src)
    imgResult = sharp - imgLaplacian

    # convert back to 8bits gray scale
    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')
    imgLaplacian = np.clip(imgLaplacian, 0, 255)
    imgLaplacian = np.uint8(imgLaplacian)

    if show_Image:
        cv2.imshow('Laplace Filtered Image', imgLaplacian)
        cv2.imshow('New Sharped Image', imgResult)

    print("Finish Laplacian and Sharping")

    # Create binary image from source image
    bw = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(bw, 80, 255, cv2.THRESH_BINARY)  # + cv2.THRESH_OTSU)
    if show_Image:
        cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Binary Image', bw)

    # cv2.imwrite("Binary Image.jpg", bw)


    print("Finish binary image and threshold")

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)

    print("Finish distance transform")

    # Normalize the distance image for range = {0.0, 1.0}
    # so we can visualize and threshold it
    cv2.normalize(dist, dist, 0.0, 1.0, cv2.NORM_MINMAX)

    print("Finish normalizing distance transform")

    if show_Image:
        cv2.namedWindow('Distance Transform Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Distance Transform Image', dist)

    # cv2.imwrite("DistanceTransform.jpg", dist)


    ## [peaks]
    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv2.threshold(dist, 0.125, 1.0, cv2.THRESH_BINARY)

    # Dilate a bit the dist image

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dist = cv2.dilate(dist, kernel)

    if show_Image:
        cv2.namedWindow('Distance Transform Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Distance Transform Image', dist)

    print("Finish dilate")

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')

    # Find total markers
    contours, hierarchy = cv2.findContours(dist_8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print("Finish finding Contours")

    # Create the marker image for the watershed algorithm
    markers = gen_marker(dist, contours)

    # Perform the watershed algorithm

    print('Start watershed')
    cv2.watershed(src, markers)
    print("Finish watershed")

    export_final_image(contours, markers)

    show_marker(markers)
    mask = np.zeros(dist_8u.shape, dtype=np.uint8)

    length = len(contours)

    print("Calculate MaxFeret of contours ...")

    feret = []
    area = []
    #  filter variable
    min_feret = 400 / pixel2um
    max_feret = 1300 / pixel2um
    max_area = 2800000 / pixel2um / pixel2um

    for i in range(length):
        mask = np.zeros(dist_8u.shape, dtype=np.uint8)
        mask[markers == i] = 255
        sp_contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(sp_contours) != 0:
            ((x, y), feret_tmp) = cv2.minEnclosingCircle(sp_contours[0])
            feret_tmp *= 2
            area_tmp = cv2.contourArea(sp_contours[0])

            # print("feret:%d, area:%d" % (feret_tmp * 2, area_tmp))

            if feret_tmp <= min_feret or feret_tmp >= max_feret or area_tmp >= max_area:
                continue

            feret.append(int(feret_tmp * pixel2um))
            area.append(int(area_tmp * pixel2um * pixel2um))
        print('Progress: %.2f %%' % (i / length * 100)),

    feret = np.array(feret)
    area = np.array(area)
    particle_info = np.vstack((feret, area))


    print("Finish max feret statistics")

    # export_final_image(contours, markers)

    # for i in range(length):
    #     ((x, y), maxFeretTmp) = cv2.minEnclosingCircle(contours[i])
    #     if maxFeretTmp < 3:
    #         max_feret.append(maxFeretTmp)  # radius to diameter

    print("Finish All Calculating")

    return np.array(particle_info)


def write_csv(max_feret_array):
    with open(r'result.csv', 'w') as file:
        for feret in max_feret_array:
            file.write("%.2f\n" % feret)


def particle_analysis(img_path):

    global show_Image
    show_Image = False

    print("img path:" + img_path)

    # TIME START
    time_start = time()

    img_src = cv2.imread(img_path)

    if img_src is None:
        print('Could not open or find the image!!')
        sys.exit(0)

    img_coffee_part, real_height, real_width = panelAbstract(img_src)

    print(real_height/img_coffee_part.shape[0]) # height

    pixel2um = real_height/img_coffee_part.shape[0]

    # cv2.imwrite("panelAbstract-Result.jpg", img_coffee_part)

    particle = get_distribution(img_coffee_part, pixel2um)

    # TIME END
    time_end = time()
    print("Spend Time: " + f'{time_end - time_start:.2f}' + 'S')

    return particle


if __name__ == '__main__':
    particle_info = particle_analysis(r"D:\Git\Coffee_project_old\edge_detector\4g_4g_4g_4g\IMG_6426.JPG")
    print(particle_info)
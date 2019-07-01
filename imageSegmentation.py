import sys
import os
import cv2
import numpy as np
import random as rng
from time import time
from imagepreprocessing import panelAbstract

show_Image = False


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
    return markers


def get_distribution(img):
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
    _, bw = cv2.threshold(bw, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if show_Image:
        cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Binary Image', bw)

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
    _, contours, hierarchy = cv2.findContours(dist_8u, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print("Finish finding Contours")

    # Create the marker image for the watershed algorithm
    markers = gen_marker(dist, contours)

    # Perform the watershed algorithm

    print('Start watershed')
    cv2.watershed(src, markers)
    print("Finish watershed")

    show_marker(markers)
    mask = np.zeros(dist_8u.shape, dtype=np.uint8)

    length = len(contours)
    max_feret = []

    print("Calculate MaxFeret of contours ...")

    for i in range(length):
        mask = np.zeros(dist_8u.shape, dtype=np.uint8)
        mask[markers == i] = 255
        test, sp_contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(sp_contours) != 0:
            ((x, y), tmp) = cv2.minEnclosingCircle(sp_contours[0])
            max_feret.append(tmp*2)
        print('Progress: %.2f %%' % (i/length*100))

    # export_final_image(contours, markers)

    # for i in range(length):
    #     ((x, y), maxFeretTmp) = cv2.minEnclosingCircle(contours[i])
    #     if maxFeretTmp < 3:
    #         max_feret.append(maxFeretTmp)  # radius to diameter

    print("Finish All Calculating")

    return np.array(max_feret)


def write_csv(max_feret_array):
    with open(r'result.csv', 'w') as file:
        for feret in max_feret_array:
            file.write("%.2f\n" % feret)

def main():
    global show_Image
    show_Image = False

    data_path = r"/home/ecl-123/zing/coffee_sever/static/images"
    all_distribution = []
    print("Path: "+data_path)
    for root_Outer, dirs_Outer, files_Outer in os.walk(data_path, topdown=False):
        for directory in dirs_Outer:
            for root, dirs, files in os.walk(os.path.join(root_Outer, directory), topdown=False):
                for name in files:

                    # TIME START
                    time_start = time()
                    img_path = os.path.join(root, name)
                    print(img_path)
                    img_src = cv2.imread(img_path)

                    if img_src is None:
                        print('Could not open or find the image!!')
                        sys.exit(0)

                    img_coffee_part = panelAbstract(img_src)
                    max_feret_array = get_distribution(img_coffee_part)

                    # TIME END
                    time_end = time()
                    print("Spend Time: " + f'{time_end - time_start:.2f}' + 'S')

                    if show_Image:
                        cv2.waitKey()
                        cv2.destroyAllWindows()

                    all_distribution.append(max_feret_array)

                array_saved = np.array(all_distribution)
                label_name = root.split('\\')[-1]

                print("Saving Data: %s.npy" % label_name)
                np.save("%s.npy" % label_name, array_saved)
                print("Finish")


if __name__ == '__main__':
    main()

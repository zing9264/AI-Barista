import cv2
import numpy as np
import math
import os
import torch
import shutil  
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from collections import OrderedDict
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d




show_Image = False
REG_OUTPUT = 8
BATCH_SIZE = 8
CUDA_DEVICES = 0
PATH_TO_WEIGHTS = '/home/ecl-123/zing/coffee_sever/Model_ResNet_Reg-8_square.pth'

class IMAGE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []  # file
        self.y = []  # index
        self.transform = transform
        self.num_classes = 0

        # print(self.root_dir.name)
        for i, file in enumerate(self.root_dir.glob('*')):

            label = [0,0,0,0,0,0,0,0]
            # label = i
            # label = [float(label), float(pow(label, 2)), float(pow(label, 3)), float(pow(label, 2)), float(label)]
            # print(f"[{os.path.split(_dir)[-1]:s}]\n\n\tLabel: {label}\n")

            self.x.append(file)
            self.y.append(torch.tensor(label, dtype=torch.float))

            self.num_classes += 1
            # print(self.num_classes)
        # print(self.num_classes)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert("RGB")
        #print(image)
        # image=image.resize((224,224))
        # image.save('transform.jpg')
        if self.transform:
            image = self.transform(image)
            # image.save('transform.jpg')
  #          print(image)
        return image, self.y[index]


def myDataloader(image_folder):

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #print(DATASET_ROOT)
    all_data_set = IMAGE_Dataset(Path(image_folder), data_transform)
    
    #print('set:',len(train_set))
    indices = list(range(len(all_data_set)))
    #print('old',indices)
    np.random.seed(1)
    np.random.shuffle(indices)
    #print('new',indices)
    split = math.ceil(len(all_data_set)*1)  # extract 10% dataset as test-set
    valid_idx = indices[:split]
    test_sampler = SubsetRandomSampler(valid_idx)
    #print('test')
    #print(test_sampler)
    #train_set, test_set = torch.utils.data.random_split(train_set, [400, 115])
    print('test_set: ',len(test_sampler))
    imgSetLen = len(test_sampler)

    test_data_loader=DataLoader(
                dataset=all_data_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                sampler=test_sampler)   
    
    return test_data_loader, imgSetLen


def test(test_data_loader, imgSetLen):

    predict_array = []
    # classes = [_dir.name for _dir in Path(DATASET_ROOT2).glob('*')]     #資料夾名稱
    model=models.resnet101()
    f=lambda x:math.ceil(x/32-7+1)
    
    # model.load_state_dict(torch.load(PATH_TO_WEIGHTS))
    model.fc=nn.Linear(f(256)*f(256)*2048, REG_OUTPUT)

    #model=nn.DataParallel(model)

    model = torch.load(PATH_TO_WEIGHTS, map_location='cpu')

    model = model.cpu()

    if isinstance(model, torch.nn.DataParallel):
         model = model.module

    # model.load_state_dict(state_dict)
    

    # model = model.cuda(CUDA_DEVICES)
    model.eval()        #test


    with torch.no_grad():

        sample_len = 0
 
        for i, (inputs, labels) in enumerate(test_data_loader):
            # inputs = inputs.cuda(CUDA_DEVICES)
            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            tmp = outputs.data.cpu().numpy()
            predictions = np.around(tmp*100)

            for predictRow in predictions:
                predict_array.append(predictRow)

            sample_len += inputs.size(0)

            print("\n================= Predictions ==================\n", predictions)

            # progress_bar_ratio = int((i+1)/len(test_data_loader)*100)

            # tmp_str = ("[Testing Progress]["+'█'*progress_bar_ratio+' '*(100-progress_bar_ratio)+"]")
            # print(tmp_str,end='\r')

        print(np.array(predict_array))

    return np.mean(np.array(predict_array), 0)


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

    cv2.imwrite("test.jpg", imgres)
    # image轉成2維，並做Threshold
    _, thresh = cv2.threshold(imgres, 127, 255, cv2.THRESH_BINARY_INV)

    threshRotate = cv2.merge([thresh, thresh, thresh])
    # 印出 threshold後的image
    # if cv2.imwrite(r"./Photo/thresh.jpg", threshRotate):
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

    print("Contours:", len(contours))

    if len(contours) > maxAreaPos:
        objCont = contours[maxAreaPos]
    else:
        print("Error: abnormal contours")
        return None  # return error code

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

        _, dstRotBW = cv2.threshold(thresh, 127, 255, 0)
        # 印出最小外接矩形

        contours, hierarchy = cv2.findContours(dstRotBW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        maxcntArea = 0
        maxAreaPos = -1
        for i in range(len(contours)):
            if maxcntArea < cv2.contourArea(contours[i]):
                maxcntArea = cv2.contourArea(contours[i])
                maxAreaPos = i
        x, y, w, h = cv2.boundingRect(contours[maxAreaPos])
        # x，y是矩陣左上角的座標，w，h是矩陣的寬與高
        # umsize代表 1pixel*umsize = 真實大小
        umsize = 90000 / w
        #   print(umsize)
        w = w / 8  # 寬度分為8等分

        # 將沒有外圍輪廓的咖啡粉存入panelImg，固定照片大小，因此h以w代替
        panelImg = srcImage[int(y + 2 * w):int(y + 6 * w), int(x +2 * w):int(x + 6 * w), :]
        # 印出圖片真實大小
        print("Image Size:", 4 * w * umsize, " um * ", 4 * w * umsize, " um")
    return panelImg


def hist_equal_lab(img):
    global show_Image

    # Converting image. to LAB Color model
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

    return cl


def preprocess(img_path, target_path,  rotateTimes):

    print("[Image Path]: ", img_path) 

    if not os.path.exists(img_path):
        print("[ERROR] Image doesn't exist !!")
        return None

    if not os.path.exists(target_path):
        print(f"[Creat directory]: {target_path}")
        os.mkdir(target_path)

    srcImage = cv2.imread(img_path)
    _, orgFilename = os.path.split(img_path)
    _, imgExtension = os.path.splitext(img_path) 
    

    (h, w) = srcImage.shape[:2]
    center = (w/2, h/2)
    error_files = []

    rotateAngle = 360/rotateTimes

    for i in range(rotateTimes):
        # 旋轉圖片
        rotation_matrix = cv2.getRotationMatrix2D(center, i*rotateAngle, 1.0)
        rstImage = cv2.warpAffine(srcImage, rotation_matrix, (w, h))

        # 切出照片中間的咖啡粉
        rstImage = panelAbstract(rstImage)

        if rstImage is None:
            error_files.append(img_path)
            break

        print("[Before]: ",rstImage.shape)

        rstImage = cv2.resize(rstImage, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        print("[After]: ", rstImage.shape)

        # CLAHE自適應直方圖均衡
        rstImage = hist_equal_lab(rstImage)

        # 印出結果
        notcut_filename = '%s_result_%d' % (orgFilename, i)
        print('notcut_Filename: ' + notcut_filename)
        print("Save in path: ", os.path.join(target_path, notcut_filename))

        for  i in range(4):
            for j in range(4):

                cutImage = rstImage[int(i*256):int((i+1)*256), int(j*256):int((j+1)*256)]
                filename = notcut_filename + '[%d][%d].%s' %(i,j, imgExtension)
                print('new_Filename: ' + filename)

                if cv2.imwrite(os.path.join(target_path, filename), cutImage):
                    print("Write Images Successfully") 

    print("Error Files:", error_files)


def predict(correct_answer = [14,26,35,11,14],imgOutputFolder = r"/home/ecl-123/zing/coffee_sever/image_preprocess/",
            imgPath = r"/home/ecl-123/zing/coffee_sever/static/images/raw_photo/Demo_input.jpg"):
    
    

    shutil.rmtree(imgOutputFolder)  

    preprocess(imgPath, imgOutputFolder, rotateTimes = 1)

    dataset, imgSetLen = myDataloader(imgOutputFolder)
    predictDist = test(dataset, imgSetLen)
    predictDist = np.around(predictDist)
    
   

    answer = [predictDist[0]+predictDist[1], 
                    predictDist[2]+predictDist[3], 
                    predictDist[4]+predictDist[5], 
                    predictDist[6], predictDist[7]]
    print("Correct-Answer: ", correct_answer)
    print("Answer: ", answer)

    font = {'family' : 'WenQuanYi Micro Hei',
        'weight' : 'normal',
        'size'   : 16}

    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(1, 1)

    x = np.array([1,2,3,4,5])
    y = np.array(answer)

    ax.grid(zorder=0)
    # plt.bar(['400-\n600','600-\n800','800-\n1000','1000-\n1100','1100-'],y, 
    #         color=['sandybrown','peru','chocolate','sienna','saddlebrown'],
    #         zorder=3) 
    plt.grid(True, linestyle = "--", which='major',color = "gray", linewidth = "1")

    index = np.arange(5)
    bar_width = 0.4

    plt.bar(['400-\n600','600-\n800','800-\n1000','1000-\n1100','1100-'],y, bar_width, alpha=.9, label="Predict", color='saddlebrown', zorder=3) 

    plt.bar(index+0.4,correct_answer,bar_width,alpha=.9,label="Answer", color='sandybrown',zorder=3) 

    plt.title("咖啡粉末分佈", fontsize=32) 
    plt.xlabel("粒徑大小(㎛)",color="black")
    plt.ylabel("重量比例",color="black")
    plt.legend()

    plt.ylim(0, 100)
    plt.tight_layout()
    
    plt.savefig("./static/images/result.jpg")

    criterion = nn.MSELoss()
    loss = criterion(torch.tensor(answer, dtype=torch.float), 
                    torch.tensor(correct_answer, dtype=torch.float))
    print(f"ALL LOSS: {loss:.2f}")
    return answer


if __name__ == "__main__":
    predict()

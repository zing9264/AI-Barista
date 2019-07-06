import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from mydataset import myIMAGE_Dataset
import torchvision.models as models
CUDA_DEVICES = 0
#DATASET_ROOT2 = '/home/pwrai/test_photo'
DATASET_ROOT3 = '/home/pwrai/new_4burr_photo'
PATH_TO_WEIGHTS = './model_pre_resnet101.pth'


def mytest():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    #data load
    test_set=myIMAGE_Dataset(Path(DATASET_ROOT3),data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=16, shuffle=False, num_workers=0)

    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()        #test
    #classes = [_dir.name for _dir in Path(DATASET_ROOT2).glob('*')]     #資料夾名稱
    classes=['5_blade','4_blade','5_burr','45_blade','4_burr']
    print(classes)
    num_class=5
    freq_list=[0 for i in range(num_class)]
    
    with torch.no_grad():
        for inputs in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)       #預測機率最高的class
            print('---predicted---')
            print(predicted.tolist())
            for i in predicted:
                freq_list[i]+=1
    
    print('---freq list---')
    print(freq_list)
    max_num=freq_list[0]
    select=0
    
    for i in range(len(freq_list)):
        if freq_list[i]>max_num:
            max_num=freq_list[i]
            select=i

    #print(max_num)
    #print(select)
    #print(classes[select])
    return(classes[select])

if __name__ == '__main__':
    result=mytest()
    print(result)

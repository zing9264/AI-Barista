import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class IMAGE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []     #file
        self.y = []     #index
        self.transform = transform
        self.num_classes = 0

        # 8
        # 400-500, 500-600, 600-700, 700-800, 800-900, 900-1000, 1000-1100, 1100-
        labels = {
                # "5_blade":[2, 22, 13, 3 , 33],
		# "45_blade":[1, 18, 11, 26, 44],
                # "4_blade":[1, 18, 6, 17, 57],
                "4_burr":[2,12,13,13,2,15,11,14],
                "4.5_burr":[14,5,8,1,16,15,14,19],
                "5_burr":[0,13,5,9,12,17,16,28],
                "400-500":[100,0,0,0,0,0,0,0],
                "500-600":[0,100,0,0,0,0,0,0],
                "600-700":[0,0,100,0,0,0,0,0],
                "700-800":[0,0,0,100,0,0,0,0],
                "800-900":[0,0,0,0,100,0,0,0],
                "900-1000":[0,0,0,0,0,100,0,0],
                "1000-1100":[0,0,0,0,0,0,100,0],
                "1100-":[0,0,0,0,0,0,100,0],
                "test_500-600_800-900_1100":[0,33,0,0,33,0,0,33],
                "test_600-700_700-800_800-900":[0,0,33,33,33,0,0,0],
                "test_900-1000_1000-1100_1100":[0,0,0,0,0,33,33,33],
                "test_1-1-1-1-3":[7,7,7,7,7,7,14,43],
                "test_1-1-2-2-1":[7,7,7,7,14,14,29,14],
                "test_2-1-1-1-1":[17,17,8,8,8,8,17,17]
                }
 
        '''
        labels = {
                "5_blade":[2, 22, 13, 3 , 33],
		"45_blade":[1, 18, 11, 26, 44],
                "4_blade":[1, 18, 6, 17, 57],
                "4_burr":[1, 25, 16, 38, 2 ],
                "5_burr":[1, 18, 12, 39, 3 ],
                "500-600":[0,100,0,0,0],
                "600-700":[0,0,100,0,0],
                "700-800":[0,0,100,0,0],
                "800-900":[0,0,0,100,0],
                "900-1000":[0,0,0,100,0],
                "1000-1100":[0,0,0,100,0],
                "1100-":[0,0,0,0,100],
                "test_500-600_800-900_1100":[0,33,0,33,33],
                "test_600-700_700-800_800-900":[0,0,66,33,0],
                "test_900-1000_1000-1100_1100":[0,0,0,66,33]}
        '''
        for key, value in labels.items():
            labels[key] = torch.tensor(value).float()*0.01

        #print(self.root_dir.name)
        for i, _dir in enumerate(self.root_dir.glob('*')):

            label = labels[os.path.split(_dir)[-1]]
            # label = i
            # label = [float(label), float(pow(label, 2)), float(pow(label, 3)), float(pow(label, 2)), float(label)]
            print(f"[{os.path.split(_dir)[-1]:s}]\n\n\tLabel: {label}\n")

            for file in _dir.glob('*'):
                #print(file)
                self.x.append(file)
                self.y.append(torch.tensor(label, dtype=torch.float))

            self.num_classes += 1
            #print(self.num_classes)
        #print(self.num_classes)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert("RGB")
 #       print(image)
        #image=image.resize((224,224))
        #image.save('transform.jpg')
        if self.transform:
            image = self.transform(image)
            #image.save('transform.jpg')
  #          print(image)
        return image,self.y[index]

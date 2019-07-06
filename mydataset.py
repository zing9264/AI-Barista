from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import os

class myIMAGE_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.x = []     #file
        #self.y = []     #index
        self.transform = transform
        self.num_classes = 0
        #print(self.root_dir.name)
        '''for i, _dir in enumerate(self.root_dir.glob('*')):
            for file in _dir.glob('*'):
                self.x.append(file)
                self.y.append(i)'''
        files= os.listdir(self.root_dir)
        #print(files)
        for file in files:
            self.x.append(str(self.root_dir)+'/'+str(file))
        self.num_classes += 1
            #print(self.num_classes)
        #print(self.num_classes)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


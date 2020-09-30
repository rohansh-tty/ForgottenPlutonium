# Change the path

# wordPath = r"C:\Users\Rohan Shetty\Desktop\S12-Assignment2\tiny-imagenet-200\tiny-imagenet-200\words.txt"

# idPath = r"C:\Users\Rohan Shetty\Desktop\S12-Assignment2\tiny-imagenet-200\tiny-imagenet-200\wnids.txt"

from torch.utils.data import Dataset, random_split
from PIL import Image
import numpy as np
import torch
import os
import torchvision.transforms as transforms
from tqdm import notebook



# pass the above libs to rohan_library

def TinyImageNetDataSet(splitRatio = 70, test_transforms = None, train_transforms = None):
  classes = xtractClassID(path = "tiny-imagenet-200/wnids.txt")
  data = TinyImageNet(classes, defPath="tiny-imagenet-200")
  
  totalLen = len(data)
  print(totalLen) # 110K

  traindata_len = totalLen*splitRatio//100 # 110K * 0.7 = 77K
  testdata_len = totalLen - traindata_len # 110K - 77K = 33K
  
  train, val = random_split(data, [traindata_len, testdata_len]) # split the data according to split ratio
  train_dataset = transformData(train, transform=train_transforms) # Data ready for Loading, passed onto Dataloader func
  test_dataset = transformData(val, transform=test_transforms)

  return train_dataset, test_dataset, classes




class TinyImageNet(Dataset):
    def __init__(self, classes, defPath):
        
        self.classes = classes
        self.defPath = defPath
        self.data = []
        self.target = []
        
        
        wnids = open(f"{defPath}/wnids.txt", "r")

        # Train Data
        trainImagePath = defPath+"/train/"
        for cls in notebook.tqdm(wnids, total = 200):
          cls = cls.strip() # strip spaces out of class names

          indFolderPath = trainImagePath + cls + "/images/"
          
          for i in os.listdir(indFolderPath): # this will list nXXXXXXXX Folders containing 500 Images.
            img = Image.open(indFolderPath + i)
            npimage = np.asarray(img)
                
            if(len(npimage.shape) == 2): 
              npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2) # add a new dim using np.newaxis, if it's a 2D
                
            self.data.append(npimage)  # appending image to data 
            self.target.append(self.classes.index(cls)) # appending corresponding class using self.classes


        # Validation Data
        valAnntns = open(f"{defPath}/val/val_annotations.txt", "r")
        for i in notebook.tqdm(valAnntns, total =10000):
          img, cls = i.strip().split("\t")[:2] # this will return image name and class ID. Ex: 'val_1.JPEG', 'n04067472'
          img = Image.open(f"{defPath}/val/images/{img}")
          npimage = np.asarray(img)
          
          if(len(npimage.shape) == 2):  
                npimage = np.repeat(npimage[:, :, np.newaxis], 3, axis=2) # add a new dim using np.newaxis, if it's a 2D
          
          self.data.append(npimage)  
          self.target.append(self.classes.index(cls))


    def __len__(self):
      """
      returns len of the dataset
      """
      return len(self.data)


    def __getitem__(self, idx):
      """
      returns image data & target for the corresponding index
      """
      data = self.data[idx]
      target = self.target[idx]
      img = data     
      return data,target
      
  

class transformData(Dataset):
    """
    Helper Class for transforming the images using albumentations.
    """
    def __init__(self, data, transform=None):
        """
        data: Train or Validation Dataset
        transform : List of Transforms that one wants to apply
        """
        self.data = data
        self.transform = transform


    def __getitem__(self, index):
        x, y = self.data[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)




def xtractClassID(path):
    """
    Helps in extracting class ID from wnids file
    """
    IDFile = open(path, "r")
    classes = []

    for line in IDFile:
        classes.append(line.strip())
    return classes


def xtractClassNames(path, wordID):
    """
    Helps in extracting ClassNames for that particular ID from words.txt file
    """
    wordFile = open(path, "r")
    classNames = {}
    # wordID = xtractClassID(idPath)

    for line in wordFile:
        wordCls = line.strip("\n").split("\t")[0] # wordCls indicates the nXXXXXXX ID
        if wordCls in wordID: 
            classNames[wordCls] = line.strip("\n").split("\t")[1]  # Adding ClassName of a particular ID(key) as a value 
    return classNames

import os
import yaml
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from collections import Counter
import torch.multiprocessing as mp
from tqdm.auto import tqdm
from chat import Lisa
import torch.nn as nn

mp.set_start_method('spawn', force=True)

def get_max_len(train, test, val):
    qtns = train["QUESTION"].to_list() + \
           test["QUESTION"].to_list() + \
           val["QUESTION"].to_list()
    
    c = 0
    for _q in qtns:
        l = len(_q.split())
        if l > c:
            c=l
    return c

class Img2MML_dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        qtn = self.dataframe.iloc[index, 1]
        img = self.dataframe.iloc[index, 0] 
        lbl = self.dataframe.iloc[index,2]
        tmp = self.dataframe.iloc[index,-1]
        
        return img,qtn,lbl,tmp
        

class My_pad_collate(object):
    def __init__(self, device, max_len):
        self.device = device
        self.max_len = max_len

    def __call__(self, batch):
        _img, _qtns, _lbls, _tmps = zip(*batch)

        # the labels will be stored as tensor
        # 3 will be stored as [0.,0.,0.,1.]
        lbls = []
        for _l in _lbls:
            _l = int(_l.replace("\n",""))
            z = torch.zeros(11)
            z[_l] = 1.0
            lbls.append(z)
        
        # tensors
        _img = torch.Tensor(_img)
        _lbls = torch.stack(lbls)

        return (
            _img.to(self.device),
            _qtns,
            _lbls.to(self.device),
            _tmps,
        )

    
def data_loaders(batch_size):

    print("creating dataloaders...")
    q = open("/home/gauravs/data/clevrmath_data/questions.lst").readlines()
    l = open("/home/gauravs/data/clevrmath_data/labels.lst").readlines()
    t = open("/home/gauravs/data/clevrmath_data/templates.lst").readlines()

    assert len(q) == len(l) == len(t)

    image_num = range(0, len(q))

    # split the image_num into train, test, validate
    train_val_images, test_images = train_test_split(
        image_num, test_size=0.1, random_state=42
    )
    train_images, val_images = train_test_split(
        train_val_images, test_size=0.1, random_state=42
    )

    for t_idx, t_images in enumerate([train_images, test_images, val_images]):
        qi_data = {
            "IMG": [num for num in t_images],
            "QUESTION": [(q[num].strip()) for num in t_images],
            "LABEL": [l[num].strip().replace("\n","") for num in t_images],
            "TEMPLATE": [t[num].strip().replace("\n","") for num in t_images],
        }
    
        if t_idx == 0:
            train = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
        elif t_idx == 1:
            test = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
        else:
            val = pd.DataFrame(qi_data, columns=["IMG", "QUESTION", "LABEL"])
    

    # get max_len 
    max_len = get_max_len(train, test, val)    
    print("the max length: ", max_len)

    # initializing pad collate class
    mypadcollate = My_pad_collate("cuda:0", max_len)

    print("building dataloaders...")

    # initailizing class Img2MML_dataset: train dataloader
    imml_train = Img2MML_dataset(train)
    # creating dataloader
    sampler = None
    shuffle = True

    train_dataloader = DataLoader(
        imml_train,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=False,
    )

    # initailizing class Img2MML_dataset: val dataloader
    imml_val = Img2MML_dataset(val)
    sampler = None
    shuffle = True

    val_dataloader = DataLoader(
        imml_val,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=mypadcollate,
        pin_memory=False,
    )

    # initailizing class Img2MML_dataset: test dataloader
    imml_test = Img2MML_dataset(test)
    sampler = None
    shuffle = True

    test_dataloader = DataLoader(
        imml_test,
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        sampler=None,
        collate_fn=mypadcollate,
        pin_memory=False,
    )

    return (train_dataloader, 
            test_dataloader, 
            val_dataloader,  
            max_len)



(
train_dataloader,
test_dataloader,
val_dataloader,
max_len,
) = data_loaders(batch_size=16)


class LisaModel(nn.Module):
    def __init__(self,):
        super(LisaModel, self).__init__()
        self.model = Lisa()

    def forward(self, imgs, qtns):
        preds, texts = self.model(imgs, qtns)
        return texts

class Adaptor(nn.Module):
    def __init__(self,):
        super(Adaptor,self).__init__()
        self.lin1 = nn.Linear(768, 512)
        self.lin2 = nn.Linear(512,256)
        self.lin3 = nn.Linear(256,128)
        self.lin4 = nn.Linear(128,64)
    
    def forward():
        pass


# >>>>>>>>>>>>>>>>>>>>>>>>>>> Train and Test <<<<<<<<<<<<<<<<<<<<<<< #

epochs = 1
device = "cuda:0"

lisamodel = LisaModel()
for param in lisamodel.parameters():
    param.requires_grad = False

adamodel = Adaptor()

for i in range(epochs):
    epoch_loss = 0

    adamodel.train()

    tset = tqdm(iter(train_dataloader))
    for i, (imgs, qtns, labels, _) in enumerate(tset):
        labels = labels.to(device, dtype=torch.long)
        texts = lisamodel(imgs,qtns)
        print(texts.shape)
        # exit()
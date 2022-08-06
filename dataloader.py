import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
import utils as ut

# PyTorch class to load the MRNet dataset

class MRDataset(data.Dataset):
    def __init__(self, root_dir, train=True, weights=None):
        super().__init__()
        self.planes = ['axial', 'coronal', 'sagittal']
        self.tasks = ['acl', 'meniscus', 'abnormal']
        self.records = {}
        # an empty dictionary
        self.image_path = {}
        self.root_dir=root_dir
        self.train= train
        if train:
            for task in self.tasks:
                self.records[task] = pd.read_csv(self.root_dir + '/train-{}.csv'.format(task), header=None, names=['id', 'label'])

            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = self.root_dir + '/train/{}/'.format(plane)
        else:
            for task in self.tasks:
                self.records[task] = pd.read_csv(self.root_dir + '/valid-{}.csv'.format(task), header=None, names=['id', 'label'])
            '''
            self.image_path[<plane>]= dictionary {<plane>: path to folder containing
                                                                image for that plane}
            '''
            for plane in self.planes:
                self.image_path[plane] = self.root_dir + '/valid/{}/'.format(plane)

        for task in self.tasks:
            self.records[task]['id'] = self.records[task]['id'].map(
                lambda i: '0' * (4 - len(str(i))) + str(i))
        # empty dictionary
        self.paths = {}
        self.labels = {}
        self.weights = []
        for task in self.tasks:
            for plane in self.planes:
                self.paths[plane] = [self.image_path[plane] + filename +
                                 '.npy' for filename in self.records[task]['id'].tolist()]

            self.labels[task]= self.records[task]['label'].tolist()

            pos = sum(self.labels[task])
            neg = len(self.labels[task]) - pos

            # Find the wieghts of pos and neg classes
            if weights:
                self.weights.append(weights[task])
            else:
                self.weights.append((neg / pos))


            print('Number of -ve samples : ', neg)
            print('Number of +ve samples : ', pos)
            print('Weights for loss is : ', self.weights)
        self.weights = torch.FloatTensor(np.asarray(self.weights))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.records[self.tasks[0]])

    def __getitem__(self, index):
        """
        Returns `(images,labels)` pair
        where image is a list [imgsPlane1,imgsPlane2,imgsPlane3]
        and labels is a list [gt,gt,gt]
        """
        img_raw = {}

        for plane in self.planes:
            img_raw[plane] = np.load(self.paths[plane][index])
            #img_raw[plane] = self._resize_image(img_raw[plane])
            if self.train:
                # data augmentation
                img_raw[plane] = ut.random_shift(img_raw[plane], 25)
                img_raw[plane] = ut.random_rotate(img_raw[plane], 25)
                img_raw[plane] = ut.random_flip(img_raw[plane])
            # data standardization
            img_raw[plane] = (img_raw[plane] - 58.09) / 49.73
            img_raw[plane] = np.stack((img_raw[plane],) * 3, axis=1)

            img_raw[plane] = torch.FloatTensor(img_raw[plane])  # array size is now [S, 224, 224, 3]
        label = []
        for key, value in self.labels.items():
            label.append(value)

        label = torch.FloatTensor(np.asarray(np.asarray(label)[:, index]))
        img = [img_raw[plane] for plane in self.planes]
        return img, label, self.weights


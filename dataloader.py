""" dataloader for images and labels """

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt

transform = transforms.Compose ({
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
})

class AssemblyDataset(Dataset):
    def __init__(self, idx1, idx2):
        # data loading
        self.image_dir = r'./data_dataset_voc/JPEGImages'
        self.label_dir = r'./data_dataset_voc/SegmentationClassPNG'

        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.label_dir)

        # self.images = self.images[idx1: idx2]
        # self.masks = self.images[idx1: idx2]

        self.transform = transform

        # self.mapping = {
        #     '_background_': 0,
        #     'left_Hand': 1,
        #     'Right_Hand': 2
        # }

        self.images.sort()
        self.masks.sort()

        # print(self.images)
        # print(self.masks)
        # lbl = np.asarray(PIL.Image.open(os.path.join(self.label_dir, self.masks[1])))
        # print(np.unique(lbl))
        # print(lbl.shape)

        

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.label_dir, self.masks[index])
        print(self.image_dir)
        print(self.images[index])

        image = np.asarray(Image.open(image_path))
        mask = np.asarray(Image.open(mask_path))

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # print("Image: ", torch.max(image))
        # print("Mask: ", mask.shape)
        print(torch.unique(mask))
        print(torch.max(torch.from_numpy(np.array(mask))))

        return image, mask
  
    def __len__(self):
        #len(dataset)
        return len(self.images)

# dataset = AssemblyDataset()

# ## Dataloader. Adjust for desired train and val sets
# dataloader = DataLoader(dataset=dataset, batch_size = 2, shuffle = True)

# ## see if images and masks are loaded correctly
# ## this adds masks on top of the image
# for i, (images, masks) in enumerate(dataloader):
#     for j in range(images.shape[0]):
#         print(f"unique values: {np.asarray(masks[j].unique())}")
#         plt.imshow(np.transpose(masks[j], (1, 2, 0)), alpha=0.6)
#         plt.show()
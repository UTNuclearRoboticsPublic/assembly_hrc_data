""" dataloader for images and labels """

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch
import numpy as np
from PIL import Image
import PIL
import os
import matplotlib.pyplot as plt

transform = transforms.Compose ([
    # transforms.Resize(size=(64, 64), interpolation=PIL.Image.NEAREST),
    transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
])

transform2 = transforms.Compose ([
    transforms.Resize(size=(161, 161), interpolation=PIL.Image.NEAREST),
    transforms.ToTensor()
])

class AssemblyDataset(Dataset):
    def __init__(self, idx1, idx2, transform2=transform2):
        # data loading
        self.image_dir = r'./data_dataset_voc/JPEGImages'
        self.label_dir = r'./data_dataset_voc/SegmentationClassPNG'

        self.images = os.listdir(self.image_dir)
        self.masks = os.listdir(self.label_dir)

        self.transform = transform
        self.transform2 = transform2

        self.images.sort()
        self.masks.sort()

        self.images = self.images[idx1:idx2+1]
        self.masks = self.masks[idx1:idx2+1]

        ### you can see the 0, 1, 2, 3 labels in the data here
        # lbl = np.asarray(PIL.Image.open(os.path.join(self.label_dir, self.masks[1])))
        # print(np.unique(lbl))
        # print(lbl.shape)

        

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.label_dir, self.masks[index])

        image = Image.open(image_path)
        image = image.convert('RGB')
        mask = Image.open(mask_path)

        # print(f"Image before transform unique: {np.unique(np.array(np.asarray(image)))}")
        # print(f"Mask before transform unique: {np.unique(np.array(np.asarray(mask)))}")


        image = self.transform2(image)
        mask = self.transform(mask)
        
        mask = torch.from_numpy(np.array(np.asarray(mask))).long()

        mask[np.where(mask==2)] = 1

        # print(f"The mask unique values are {mask.unique()}")


        # print(f"Image after transform unique: {torch.unique(image)}")
        # print(f"Mask after transform unique: {torch.unique(mask)}")

        # print(f"Image after transform shape: {image.shape}")
        # print(f"Mask after transform shape: {mask.shape}")

        return image, mask
  
    def __len__(self):
        return len(self.images)
    

# """Uncomment to test and view samples from dataset"""
    
# dataset = AssemblyDataset(0, 2)

# # # # ## Dataloader. Adjust for desired train and val sets
# dataloader = DataLoader(dataset=dataset, batch_size = 2, shuffle = True)

# # see if images and masks are loaded correctly
# # this adds masks on top of the image
# for i, (images, masks) in enumerate(dataloader):
#     for j in range(images.shape[0]):
#         # print(f"unique values: {np.asarray(masks[j].unique())}")
#         # print(f"shape of image {images[j].shape}")
#         # plt.imshow(images[j].permute(1, 2, 0))
#         plt.imshow(np.transpose(images[j])) # , (1, 2, 0)
#         plt.savefig("image.jpg")
#         # print(np.transpose(masks[j]).dtype)  # , (1, 2, 0)
#         folder="saved_images/" 
#         # torchvision.utils.save_image(np.transpose(masks[j], (1, 2, 0)).astype(np.uint8), f"{folder}{j}.png")
#         # image = Image.fromarray(np.uint(np.transpose(masks[j], (1, 2, 0))))
#         plt.imshow(np.transpose(masks[j]), alpha=0.6)
#         image = np.transpose(masks[j]) # , (1, 2, 0)
#         plt.show()
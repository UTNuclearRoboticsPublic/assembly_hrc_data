# assembly_hrc_data
Labelled data from HRC experiments to be used for semantic hand segmentation. Includes ability to train using baseline UNET architecture. 

![sample](samplevisual.png)

## Branch Structure for FIRE
All of the current work is currently in the `shondle/with-egohands` branch, although not all of the code works.

Working additions are in main branch, but this is not the most updated code. Most of other branches work has been merged into the `shondle/with-egohands` branch, and if not, it is experimental.  

## Training model

Run `train.py`. 

At the top of `train.py` you can find parameters to adjust for training. Most relevant, you can set NUM_NETS to greater than 1 if you want to use deep ensembles, and can choose the model architecture you want to use. Under that, you may adjust the dataset you want to use (both for training and testing) as shown in the file. *Currently there is no functionality for using the EgoHands dataset, as the repo with that needs to be added as a submodule. This feature will be added soon.*


## Converting LabelMe Annotations to Trainable Dataset
Enter the following command into anaconda prompt in the project directory-
(where images and JSON annotations are under Labelled/train/images)
`./labelme2voc.py ./Labelled/train/images data_dataset_voc --labels labels.txt`

Setup environment-
`conda env create -f environment.yml`

Then, visualize data by running `dataloader.py`. Adjust for use case.

## Credits
labelme2voc.py from https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation


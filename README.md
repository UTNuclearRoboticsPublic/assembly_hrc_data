# assembly_hrc_data
Labelled data from HRC experiments to be used for semantic hand segmentation. Includes ability to train using baseline UNET architecture. 

![sample](samplevisual.png)

## Setup environment-

`conda env create -f config/environment.yml`

## Setting up submodule
1. `git submodule update --init --recursive`
2. Change into the `EgoHands` directory
3. `git checkout shondle/modulewithassembly`

## Training model

Run `train.py`. Queried dataset can be adjusted in `utils.py`.

## Viewing Tensorboard output
`git submodule update --init --recursive`

## Converting LabelMe Annotations to Trainable Dataset
Enter the following command into anaconda prompt in the project directory-
(where images and JSON annotations are under Labelled/train/images)
`./labelme2voc.py ./Labelled/train/images data_dataset_voc --labels labels.txt`

## Credits
labelme2voc.py from https://github.com/wkentaro/labelme/tree/main/examples/semantic_segmentation


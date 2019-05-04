# Color Splash Example

This is an example showing the use of Mask RCNN in a real application.
We train the model to detect fish only, and then we use the generated 
masks to keep fish in color while changing the rest of the image to
grayscale.


## Installation
1. Download `mask_rcnn_fish.h5`. Save it in the root directory of the repo (the `mask_rcnn` directory).
2. Download `fish_dataset.zip`. Expand it such that it's in the path `mask_rcnn/datasets/fish/`.

## Apply color splash using the provided weights
Apply splash effect on an image:

```bash
python3 fish.py splash --weights=/path/to/mask_rcnn/mask_rcnn_fish.h5 --image=<file name or URL>
```

Apply splash effect on a video. Requires OpenCV 3.2+:

```bash
python3 fish.py splash --weights=/path/to/mask_rcnn/mask_rcnn_fish.h5 --video=<file name or URL>
```


## Run Jupyter notebooks
Open the `inspect_fish_data.ipynb` or `inspect_fish_model.ipynb` Jupter notebooks. You can use these notebooks to explore the dataset and run through the detection pipelie step by step.

## Train the Balloon model

Train a new model starting from pre-trained COCO weights
```
python3 fish.py train --dataset=/path/to/fish/dataset --weights=coco
```

Resume training a model that you had trained earlier
```
python3 fish.py train --dataset=/path/to/fish/dataset --weights=last
```

Train a new model starting from ImageNet weights
```
python3 fish.py train --dataset=/path/to/fish/dataset --weights=imagenet
```

The code in `fish.py` is set to train for 3K steps (30 epochs of 100 steps each), and using a batch size of 2. 
Update the schedule to fit your needs.

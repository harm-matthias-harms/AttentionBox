# AttentionBox

AttentionBox: Efficient Object Proposal Generation based on [AttentionMask](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people-alt/wilms/attentionmask.html). ([Bachelor Thesis](https://drive.google.com/file/d/1jqt-g4p6NDFy-xRlNABv9gpeUwdV3MUr/view?usp=sharing))

In my bachelor thesis, I evaluated [AttentionMask](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people-alt/wilms/attentionmask.html) with a box regressor replacing the segmentation module. With AttentionBox, I propose an efficient method generating object proposals with an average performance. AttentionBox decreases the runtime of AttentionMask by 32% because the boxes don't need to be derived from masks anymore. While running at 7fps, AttentionBox has a decent performance detecting small and tiny objects.

![Example](/example.png)

The system is based on [AttentionMask](https://www.inf.uni-hamburg.de/en/inst/ab/cv/people-alt/wilms/attentionmask.html), which is based on [FastMask](https://arxiv.org/abs/1612.08843).

# Results

All methods were evaluated on boxes.

| Method            | AR@10   | AR@100  | AR@1k   | AR<sup>S</sup>@100     | AR<sup>M</sup>@100     | AR<sup>L</sup>@100     | time    |
|---------------    |-------  |-------- |-------  |--------------------    |--------------------    |--------------------    |-------  |
| BING              | 0.037   | 0.084   | 0.163   | -                      | -                      | -                      | 0.20s   |
| EdgeBoxes         | 0.074   | 0.178   | 0.338   | 0.017                  | 0.0138                 | 0.505                  | 0.31s   |
| MCG               | 0.101   | 0.178   | 0.398   | -                      | -                      | -                      | 45s     |
| DeepMaskZoom      | 0.191   | 0.378   | 0.511   | 0.141                  | 0.493                  | 0.617                  | 1.35s   |
| SharpMask         | 0.198   | 0.367   | 0.490   | 0.063                  | 0.514                  | 0.674                  | 1.03s   |
| SharpMaskZoom     | 0.202   | 0.397   | 0.533   | 0.147                  | 0.519                  | 0.648                  | 2.02s   |
| FastMask          | 0.227   | 0.430   | 0.568   | 0.175                  | 0.549                  | 0.692                  | 0.33s   |
| AttentionMask     | 0.214   | 0.426   | 0.570   | 0.210                  | 0.508                  | 0.673                  | 0.22s   |
| AttractioNet      | 0.326   | 0.532   | 0.660   | 0.317                  | 0.621                  | 0.771                  | 1.63s   |
| ZIP               | 0.335   | 0.539   | 0.612   | 0.319                  | 0.630                  | 0.785                  | 1.13s   |
| AttentionBox      | 0.219   | 0.429   | 0.560   | 0.219                  | 0.525                  | 0.649                  | 0.15s   |

# Demo

First 5k images of the MS COCO val2014 for different methods. The images show whether an object was found or not:

[AttentionBox](https://drive.google.com/open?id=1dUOInJX0-rDkrLXVvW7WvuWe9wBIdh1a), [AttentionMask](https://drive.google.com/open?id=1zLDhFp28-Jy0hTyguEVW7COElkmCgP7Q), [FastMask](https://drive.google.com/open?id=1Fezyncv-7BhbsWnHC52RzqIlk7wstiOF), [AttractioNet](https://drive.google.com/open?id=1WVd0lWmSMmu5M6JYli6lI6OVIiYDbaZP)

Results.json for the first 5k images of the MS COCO val2014 for different methods:

[AttentionBox](https://drive.google.com/uc?id=1-ZyEGFQS__p-fEGvAPgsNaH3LNFwnyR1), [AttentionMask](https://drive.google.com/uc?id=1sL6BfK6Psd293n4cQN1doxZd-yzZ_XAr), [FastMask](https://drive.google.com/uc?id=13SQ4VifyEW2YhZXzynQVpRCwTHGc4NxR), [AttractioNet](https://drive.google.com/uc?id=1G65MEJ4VrdNhfzeilh20RvLRClmW7CCj)

Tipp: Download files from Google Drive using [gdown](https://pypi.org/project/gdown/)

# Installation

The following is adapted from [AttentionMask's Repository](https://github.com/chwilms/AttentionMask).

## Requirements

- Ubuntu 16.04
- Cuda 9.0
- Python 2.7
- OpenCV-Python
- Python packages: scipy, numpy, python-cjson, setproctitle, scikit-image
- [COCOApi](https://github.com/pdollar/coco)
- Caffe (already part of this git)
- [Alchemy](https://github.com/voidrank/alchemy) (already part of this git)

## Hardware specifications

For the results of this thesis, I used the following hardware:

- Intel i7-5930K 6 core CPU
- 32 GB RAM
- GTX Titan X GPU with 12 GB RAM

## Installation

I assume Ubuntu 16.04 with Cuda 9.0, Python 2.7 and pip already installed.

First, install OpenCV-Python:

```bash
sudo apt-get install python-opencv
```

Then, clone and install COCOApi as described [here](https://github.com/pdollar/coco).

Now, clone this repository, install the Python packages from `requirements.txt` if necessary, and install the requirements of Caffe (PyCaffe) following [the official instructions](https://caffe.berkeleyvision.org/installation.html). Edit the `Makefile.config` according to your system settings.

```bash
git clone https://github.com/harm-matthias-harms/AttentionBox
cd AttentionBox
pip install -r requirements.txt
cd caffe
make pycaffe -j6
cd ..
```

Create new subdirectories for weights `params` and results `results`:

```bash
mkdir params results
```

## Usage

After successful installation, AttentionBox can immediately be used without any training. Just download the weights (and the COCO dataset). However, you can also train AttentionBox with your own data. Note, however, that your own data should be in the COCO format.

### Download dataset

Download the `train2014` and `val2014` splits from [COCO dataset](https://cocodataset.org/#download). The `train2014` split is exclusively used for training, while the first 5000 images from the `val2014` split are used for testing. After downloading, extract the data in the following structure:

```
AttentionBox
|
---- data
     |
     ---- coco
          |
          ---- annotations
          |    |
          |    ---- instances_train2014.json
          |    |
          |    ---- instances_val2014.json
          |
          ---- train2014
          |    |
          |    ---- COCO_train2014_000000000009.jpg
          |    |
          |    ---- ...
          |
          ---- val2014
               |
               ---- COCO_val2014_000000000042.jpg
               |
               ---- ...
```

### Download weights

For inference, you have to download the model weights for the final AttentionBox model: [AttentionBox-final](https://drive.google.com/uc?id=1cYIo5qiwrylM9rldFwNeQaS83PYY9j3T).

If you want to do the training yourself, download the [initial ImageNet weights for the ResNet](https://drive.google.com/uc?id=1lzO19yAwm3Ovaip1O7S6sAXCUvX8egCN). Weight files should be moved into the `params` subdirectory.

Tipp: Download files from Google Drive using [gdown](https://pypi.org/project/gdown/)

### Inference

There are two options for inference. You can either generate proposals for the COCO dataset (or any other dataset following that format) or you can generate proposals for one image.

#### COCO dataset

For inference on the COCO dataset, use the `testAttentionBox.py` script with the GPU id, the model name, the weights and the dataset you want to test on (e.g., val2014):

```bash
python testAttentionBox.py 0 attentionBox-final --init_weights attentionBox-final.caffemodel --dataset val2014 --end 5000
```

By default, only the first 5000 images of a dataset are used.

#### Individual image

If you want to test AttentionBox on one of your images, call the `demo.py` script with the path to your image, the GPU id, the model name, and the weights:

```bash
python demo.py 0 attentionBox-final <your image path here> --init_weights=attentionBox-final.caffemodel
```

As a result, you get an image with the best 20 proposals as overlays. If you want to dive deeper into the set of proposals, you can store them all using the `ret_masks` variable in the script with the `ret_scores` variable for the objectness scores.

### Evaluation

For the evaluation on the COCO dataset, you can use the `evalCOCO.py` script with the model name and the dataset used. `--useSegm` is a flag for using segmentation masks instead of bounding boxes.

```bash
python evalCOCO.py attentionBox-final --dataset val2014 --useSegm False --end 5000
```

By default, only the first 5000 images of a dataset are used.

### Training

To train AttentionBox on a dataset, you can use the `train.sh` script. It iterates over several epochs and saves as well as evaluates the result of each epoch (outputs in `trainEval.txt`). For validation, currently, the first 5000 images of the training set are used. However, we encourage you to use a split of `val2014` that is disjunct to the first 5000 images as your own validation set. Lowering the learning rate has to be done manually. We lowered the learning rate after three consecutive epochs of not improving results on our validation set. The environmental variable `EPOCH` determines the next epoch to be run and is automatically incremented.

```bash
export EPOCH=1
./train.sh
```

#### Training on your own dataset

If you want to change the dataset form COCO to something else you have to follow the subsequent steps.

1. You have to provide the annotations in the COCO-style. COCO-style means the annotation file has to be a JSON file similar to the COCO annotations. There are many tools on the web to change or create annotations accordingly.

2. Change the `shuffledData.txt`. The only purpose of this file is to keep the data preprocessing in the data layer and the box selection layer in sync (loading the identical image, determining if the image should be flipped or slightly zoomed). Therefore this file keeps a randomly shuffled list of all indices of the dataset. In the case of the COCO dataset, it is a list of numbers from 0 to 82080 (82081 images in training set). Additionally, for each number, there is a random flag (0 or 1) for horizontally flipping the image for training as well as a number between 0 and 69 as a tiny zoom. A tiny zoom is a small number that is added on top of the max edge length to get some more variety in image sizes (check `fetch()` and `fetch_image()` in `base_coco_ssm_spider.py` or `boxSelectionLayerMP.py` for details). All three values are separated by a semicolon, and each line has one entry.

3. In `config.py` adjust `ANNOTATION_TYPE` and `IMAGE_SET` according to your new dataset. Furthermore, you may have to adjust `ANNOTATION_FILE_FORMAT` for the path to the annotations or `IMAGE_PATH_FORMAT` for the path to the images. The image format strings are used in `alchemy/data/coco.py` for locating the images. Changes may have to be applied there as well, e.g., if the image file name does not start with the dataset name.

4. The solver (`models/*.solver.prototxt`) has to be adapted if the dataset is of a different length than COCO. Change snapshot, display, and average_loss according to the number of images in your dataset.

5. Change the value of the `--step` parameter when calling the training script to the number of images in your dataset.

6. For inference, you may have to change the extraction of `image_id` in the `testAttentionBox.py` script according to the image IDs in your dataset.

### Speed up training or testing / Decrease network size

To speed up the training or testing phase as well as decreasing the network's memory footprint, removing one or multiple scales is a straightforward solution. However, this comes at the cost of (slightly) decreased performance in terms of AR. Usually removing scales `24`, `48`, and `96` results only in little changes in terms of AR. Removing scale `8` on the other hand results in large gains in terms of speed up and memory footprint. However, it decreases the performance on small objects significantly. For removing one or multiple scales, follow the subsequent steps.

#### Testing

Usually, it is no problem to test with fewer scales than you used for training. We will show here how to remove scale `128` from `myModel`. Removing other scales works accordingly.

1. From the `configs/myModel.json` file remove `128` from the list `RFs`.
2. In the `models/myModel.test.prototxt` remove the following elements:
   1. Remove `top: "sample_128"` from `sample_concat layer`.
   2. Remove the `layers extractor_128` and `conv_feat_1_128s`.
   3. Remove `top: "obj128_flags"` and `bottom: "obj128_checked"` from the `SplitIndices` layer.
   4. Remove `bottom: "obj128_checked"` and `bottom: "nonObj128_checked"` from `concatFlattendObj` and `concatFlattendNonObj` layer, respectively.
   5. Remove the entire `objectness attention at scale 128` block. This step is not necessary for scale `8`, scale `16` and scale `24`.
   6. Remove the entire `shared neck at scale 128` block. This step is only necessary, if you remove a final scale from one of the branches (e.g. scale `96` or scale `128` from `attentionBox-final`). This step is not necessary for scale `8`, scale `16` and scale `24`.

#### Training

1. Follow the instructions for testing.
2. Remove all elements in the `models/myModel.train.prototxt` file that have been removed from the `models/myModel.test.prototx` file.
3. Furthermore, remove `top: "objAttBox_128"` and `top: "objAttBox_128_org"` from the `data` layer.
4. Remove `128` from the `attr` in `coco/coco_ssm_spider.py`.

#### Removing an entire branch

If you remove scales `24`, `48`, `96`(, `192`), you can also remove the `div3 branch` from the base net (layers marked with `_div3`).

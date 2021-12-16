# VRDL HW3 - Nuclei Segmentation
###### tags: `基於深度學習之視覺辨識專論`
## Introduction
![](https://i.imgur.com/RaOLP2s.jpg)
In this assignment, we are given we are given 24 training images and 6 test images. Each image contains multiple nucleus. Our goal is to train an instance segmentation model to detect and segment all nuclei in the image.

I use [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) as my code base. In this repository, I include the main code `myNucleus.py` only. You need to install [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) first and then paste my code into it. See Installation part for more detail.

## Environment
### Hardware
* CPU: Intel i5-7500 CPU
* GPU: NVIDIA GeForce GTX 1060 6GB

## Installation
1. Refer to the installaton instructions on [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN) project. The following are the installation steps based on my environment FYI.
    ```
    conda create --name TF(1.15) python=3.7

    conda install cython
    pip install opencv-python
    conda install -c anaconda pillow
    conda install -c anaconda scikit-image
    conda install imgaug


    去NVIDIA裝CUDA10.0 & 拖曳cudnn到bin/include/lib資料夾
    conda install cudatoolkit=10.0 
    conda install cudnn=7.6.5 

    pip install h5py==2.10.0
    pip install tensorflow==1.15.0
    pip install tensorflow-gpu==1.15.0
    pip install keras==2.1.6


    裝pycocotools
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

    cd進Mask_RCNN，執行
    python setup.py install

    conda install -c anaconda ipykernel (最後再裝)
    (在VS code中遇到ipykernel無法執行，先conda uninstall traitlets，再重新裝ipykernel)
    ```
    If everything goes fine, you should be able to run [samples/demo.ipynb](https://github.com/matterport/Mask_RCNN/blob/master/samples/demo.ipynb) successfully.

2. Put `myNucleus.py` into `Mask_RCNN/samples/nucleus` folder.


## Prepare dataset
1. Create a folder named `myDataset`.
2. Create folder `myTrain` and `myTest`, then put training data and testing data into it.

Organize training like this:
![](https://i.imgur.com/3l88gdo.png)

Organize testing data like this:
![](https://i.imgur.com/qCDCYww.png)
![](https://i.imgur.com/fu8tlom.png)


## Train & Test
Make sure you activate the conda environment, and cd to `Mask_RCNN/samples/nucleus` first.

### Train
Run this command to train from COCO pre-trained weights.
```
python myNucleus.py train --dataset=myDataset --subset=myTrain --weights=coco
```

### Test (Inference)
To test the model and generate `answer.json`, create a folder named `output` and put [mask_rcnn_nucleus_0050.h5](https://drive.google.com/file/d/16GC3c6XLKSu5lI4N7aaP7WLqUhWpHMhb/view?usp=sharing) inside. Then, run this command:
```
python myNucleus.py detect --dataset=myDataset --subset=myTest --weights=output/mask_rcnn_nucleus_0050.h5
```
After execution, it will generate `answer.json` file under `samples/nucleus`.



## Results
(Refer to the report for more details.)

| Model Name  | Backbone | Best score on CodaLab |
| :-: | :-: | :-: |
| [mask_rcnn_nucleus_0050.h5 ](https://drive.google.com/file/d/16GC3c6XLKSu5lI4N7aaP7WLqUhWpHMhb/view?usp=sharing)  | ResNet50 | 0.244301  |

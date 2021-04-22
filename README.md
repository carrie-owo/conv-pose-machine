# conv-pose-machine
This is the personal **TensorFlow** implementation of *Convolutional Pose Machines*, CVPR'16.

## Datasets
[LSPET Dataset](https://sam.johnson.io/research/lspet.html) and [LSP Dataset](https://sam.johnson.io/research/lsp.html) is used. The number of key points is 14.

## Environment
The code is developed using Python 3.7 on Windows 10. NVIDIA GPUs are not requirement, but we suggest using. The code is developed and tested using NVIDIA GeForce RTX3080 Laptop.

## Virtual Environment on A Personal Machine
To work on a personal machine, we can recreate the env locally. For this, we will use the venv utility to create ourselves a new Python virtual environment. First, make sure that you have the correct version of Python installed. You can download version 3.7.
### On macOS and Linux
To check if Python 3.7 is installed, on Linux and macOS we can run:
```shell
$ python3.7 --version
```


## Training CPM
- First download dataset and modify the path of dataset in **train.py**.
- You can adjust training parameters to fit your own dataset, such as the number epoch of training process and so on.
- Train CPM model
  ```shell
  python train.py
  ```
  
## Test
You can download some images to test this CPM model. The size of image is not fixed. After downloading, modify the the path of test image in **test.py**.
To show running results:
```shell
python test.py
```
Then the model estimation result will be shown and saved.

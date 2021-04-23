# conv-pose-machine
This is the personal **TensorFlow** implementation of *Convolutional Pose Machines*, CVPR'16.
## Datasets
[LSPET Dataset](https://sam.johnson.io/research/lspet.html) and [LSP Dataset](https://sam.johnson.io/research/lsp.html) is used. The number of key points is 14.
## Environment
The code is developed using Python 3.7 on Windows 10. NVIDIA GPUs are not requirement, but we suggest using. The code is developed and tested using NVIDIA GeForce RTX3080 Laptop.
## Requirement
Let's use **requirements.txt** to install all needed libraries using the following command on your local computer:
```shell
(env) $ pip install -r requirements.txt
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
## Result
When we train our model, we used 1 batch and 70 epochs. In the test, we got the heatmap corresponding to 14 key points

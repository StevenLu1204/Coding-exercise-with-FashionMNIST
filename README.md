# Coding exercise with FashionMNIST
See how basic image processing affect inferencing quality.

## Training

The model was constructed with MobileNet and ImageNet pretrained weights. You can train the model with:

`python train.py`

## Inferencing

You can run the experiment with:

`python infer.py`

The prediction accuracy on the original testing data was 93.67%. After some basic image processing (complementary value, Gaussian blur) on the testing images, the accuracy decreases.

It can be seen that both process have a huge influence on the CNN.

### Accuracy vs percentage of the pixels converted to complementary value
![Accuracy vs percentage of the pixels converted to complementary value](com_acc.png)

### Accuracy vs std of the Gaussian 3x3 kernel
![Accuracy vs std of the Gaussian 3x3 kernel](gblur_acc.png)



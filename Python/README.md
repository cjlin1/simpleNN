# SimpleNN_Tensorflow

A Tensorflow implementation of SimpleNN with GPU supported. 

# Requirements
Python dependecies:
```
python 3.5.2
tensorflow 1.13.1
tensorboard
numpy
scipy
```

# Train
We provide exactly the same CIFAR10 dataset used in MATLAB part to test our codes. **Essentially, we do not feed the whole samples into GPU memory to evaluate sub-sampled Gauss-Newton matrix at once.** Instead we divide the samples into segment of size **bsize** and take average of them to avoid out-of-memory issue.

For your own dataset, you may want to rewrite the **read_data** function in the **utilities.py** file which returns tuple **(data, labels)** in numpy format. 

- **data**: (NUM_OF_DATA × DIMENSION_OF_DATA)
- **labels**: (NUM_OF_DATA × NUMBER_OF_CLASS)

If you want to rewrite our model, the model needs to return a tuple **(x, y, outputs)**. (Note: batchnorm layer has not been fully understood and supported yet)

- **x**: placeholder for input 
- **y**: placeholder for label
- **outputs**: output value of the neural network (pre-softmax layer)

## Examples
To use Newton optimizer, please run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --optim NewtonCG --s 5000 --C 0.01  \
						--net CNN_6layers --bsize 1024
```
To use SGD optimizer, please run:
```
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --optim SGD --lr 0.01 --C 0.01 \
						--net CNN_6layers --bsize 256
```

## Arguments
In this section, we show option/parameters that are solely for Tensorflow implementation. The remaining arguments are maintained the same as the MATLAB version of [SimpleNN](https://github.com/cjlin1/simpleNN). Please refer to [SimpleNN](https://github.com/cjlin1/simpleNN) for details. The sample usage given below also indicates the default value of each parameter.

### General
1. **--optim**: the optimization method used for training CNN. (NewtonCG, SGD or Adam)
```
--optim NewtonCG
```
2. **--net**: model that we provide (CNN_3layers, CNN_6layers)
```
--net CNN_3layers
```
3. **--train_set** & **--val_set**: provide the directory of .mat file for training or validation.
```
--train_set data/cifar10.mat
```
4. **--model**: save log and model to directory log_name
```
--model ./log_and_model/logger.log
```
5. **--loss**: which loss function to use: MSELoss or CrossEntropy
```
--loss: MSELoss
```
6. **--bsize**: Split data into segements of size **bsize** so that they can fit into memroy, while evaluating Gv, stochastic gradient and global graident. If you encounter Out of Memory (OOM) during training, you may decrease the **--bsize** paramters to an appropriate value.
```
--bsize 1024;
```
7. **--print_log_only**: screen printing running log instead of storing it
```
--print_log_only
```
8. **--C**: regularization term, or so-called weight decay where weight_decay = lr/(C × num_of_samples). In this implementation, regularization term = 1/(2C × num_data) × L2_norm(weight)
```
--C math.inf
```

### Newton Method

1. **--s**: number of samples used in the subsampled Gauss-Newton matrix.
```
--s 5000
```

### SGD
1. **--decay**: divide learning rate by 10 every **decay** epochs
```
--decay 500
```

# Predict
You may need the following arguments to run the predict script:
1. **--model**: provide the address of test model, i.e. ./log_and_model/best-model.ckpt
```
--model ./log_and_model/best-model.ckpt
```
2. **--net**: model to be tested
```
--net CNN_3layers
```
3. **--test_set**: provide the directory of .mat file for test.
```
--test_set data/cifar10.t.mat
```

## Example
```
CUDA_VISIBLE_DEVICES=$GPU_ID python predict.py --net CNN_3layers --bsize 1024 \									--test_set data/cifar10.t.mat \
						--model ./log_and_model/best-model.ckpt
```

# Experiment Results

In the following experiments, we run 100 Newton steps on Newton method and 500 epochs on SGD. We report our resutls on both 3-layer and 6-layer CNN with MSE loss function. We consider the same 3-layer CNN setting in [Wang et al.](https://www.csie.ntu.edu.tw/~cjlin/papers/cnn/newton-CNN.pdf). Other settings such as the initialization are also the same as [Wang et al.](https://www.csie.ntu.edu.tw/~cjlin/papers/cnn/newton-CNN.pdf) for both 3-layer and 6-layer CNN. Both netowrks are trained and tested on CIFAR10 dataset.

## Experiments on 3 layer CNN

### Wall clock time comparison
![](./images/ACCU_TIME_3_layers.png "accu vs time")

![](./images/loss_3layers.png "loss of 3 layer CNN")

![](./images/accu_3layers.png "accuracy of 3 layer CNN")

### Test accuracy changes w.r.t. regularization constant.
C | 10% sub-sampled Gv| 5% sub-sampled Gv| 1% sub-sampled Gv
--|-----|----|---
0.01l|78.09%|78.33%|75.62%
0.1l|74.96%|75.33%|73.45%
1l|73.03%|73.35%|72.82%

Memory | bsize 1024 | bsize 512| bsize 256
--|-----|----|---
10% sub-sampled Gv|3.1 GB |1.8 GB|1.1 GB
5% sub-sampled Gv |3.1 GB |1.8 GB|1.1 GB
1% sub-sampled Gv |3.1 GB |1.8 GB|1.1 GB
SGD |3.1 GB|1.8 GB|1.1 GB|

## Experiments on 6 layer CNN
### Wall clock time comparison
![](./images/ACCU_TIME_7_layers.png "accu vs time")

![](./images/loss_7layers.png "loss of 6 layer CNN")

![](./images/accu_7layers.png "accuracy of 6 layer CNN")

### Test accuracy changes w.r.t. regularization constant.
C | 10% sub-sampled Gv| 5% sub-sampled Gv| 1% sub-sampled Gv
--|-----|----|---
0.01l|80.61%|81.41%|75.50%
0.1l|74.06%|73.55%|75.90%
1l|71.03%|70.83%|76.29%

Memory | bsize 1024 | bsize 512| bsize 256
--|-----|----|---
10% sub-sampled Gv|7.2 GB |3.8 GB|2.1 GB
5% sub-sampled Gv |7.2 GB |3.8 GB|2.1 GB
1% sub-sampled Gv |7.2 GB |3.8 GB|2.1 GB
SGD |7.2 GB|3.8 GB|2.1 GB|
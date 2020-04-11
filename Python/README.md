# SimpleNN_Tensorflow

A Tensorflow implementation of SimpleNN with GPU support. 

# Requirements
Python dependecies (Tensorflow 1.13, 1.14, 1.15 and 2.0 are now **supported**!):
```
python 3.5.2
tensorflow 2.0
```

# Train
**We do not feed the whole subsampled data into GPU memory to evaluate sub-sampled Gauss-Newton matrix vector product.** Instead we divide the samples into segments of size **bsize** and accumulate results to avoid the out-of-memory issue. For the core operation of Gauss-Newton matrix-vector products, we use Tensorflow's vector-Jacobian products; see implementation details in this [document](https://www.csie.ntu.edu.tw/~cjlin/papers/cnn/Calculating_Gauss_Newton_Matrix_Vector_product_by_Vector_Jacobian_Products.pdf).

If a validation set is provided, the program gets the validation accuracy at each iteration and returns the best model. If a validation set is not provided, then the model obtained at the last iteration is returned. For the data format, currently we assume the same format as the **MATLAB** code. See details in the README file of the **MATLAB** directory. The **read_data** function in the **utilities.py** will read **MATLAB** file, perform data normalization and reshape the input data. Please make sure the input data is in **MATLAB** format, and each input instance is vectorized.

input:

- **data**: (NUM_OF_DATA, DIMENSION_OF_DATA)
- **labels**: (NUM_OF_DATA,)

return:
- **data**: (NUM_OF_DATA, HEIGHT, WIDTH, CHANNEL)
- **labels**: (NUM_OF_DATA, NUM_OF_CLASSES)


<!-- For your own dataset, you may want to rewrite the **read_data** function in the **utilities.py** file which returns tuple **(data, labels)** in numpy format. 

- **data**: (NUM_OF_DATA × DIMENSION_OF_DATA)
- **labels**: (NUM_OF_DATA × NUMBER_OF_CLASS)

If you want to rewrite our model, the model needs to return a tuple **(x, y, outputs)**. (Note: batchnorm layer has not been fully understood and supported yet)

- **x**: placeholder for input 
- **y**: placeholder for label
- **outputs**: output value of the neural network (pre-softmax layer) -->

## Examples
To use Newton optimizer, please run:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --optim NewtonCG --GNsize 100 --C 0.01  \
                                                --net CNN_4layers --bsize 1024 \
                                                --train_set ./data/mnist-demo.mat \
                                                --val_set ./data/mnist-demo.t.mat --dim 28 28 1
```
To use SGD optimizer, please run:
```
CUDA_VISIBLE_DEVICES=0 python3 train.py --optim SGD --lr 0.01 --C 0.01 \
                                                --net CNN_4layers --bsize 256 \
                                                --train_set ./data/mnist-demo.mat \
                                                --val_set ./data/mnist-demo.t.mat --dim 28 28 1
```

## Arguments
In this section, we show option/parameters that are solely for the Python implementation. **For other arguments see details in README of the MATLAB directory of [SimpleNN](https://github.com/cjlin1/simpleNN/tree/master/MATLAB).**

### General
1. **--optim**: the optimization method used for training CNN. (NewtonCG, SGD or Adam)
```
Default: --optim NewtonCG
```
2. **--net**: network configuration (CNN_4layers, CNN_7layers, VGG11, VGG13, VGG16, and VGG19)
```
Default: --net CNN_4layers
```
3. **--train_set** & **--val_set**: provide the address of .mat file for training or validation (optional). 
```
Default: --train_set data/mnist-demo.mat
```
4. **--model**: save model to a file
```
Default: --model ./saved_model/model.ckpt
```
5. **--loss**: which loss function to use: MSELoss or CrossEntropy
```
Default: --loss MSELoss
```
6. **--bsize**: Split data into segements of size **bsize** so that each segment can fit into memroy for evaluating Gv, stochastic gradient and global graident. If you encounter Out of Memory (OOM) during training, you may decrease the **--bsize** paramter to an appropriate value.
```
Default: --bsize 1024
```
7. **--log**: saving log to a file
```
Default: --log ./running_log/logger.log
```
8. **--screen_log_only**: if specified, log printed on screen only but not stored to the log file
```
Default: --screen_log_only
```
9. **--C**: regularization parameter. Regularization term = 1/(2C × num_data) × L2_norm(weight)^2
```
Default: --C 0.01
```
10. **--dim**: input dimension of data. Shape must be: height width num_channels
```
Default: --dim 32 32 3
```
11. **--seed**: specify random seed to make results deterministic. If no random seeds are given, a different result is produced after each run.
```
Default: --seed 0
```
12. **--profile**: enable profiling
```
Default: false
```

### Newton Method

1. **--GNsize**: number of samples used in the subsampled Gauss-Newton matrix.
```
Default: --GNsize 4096
```

### SGD
1. **--decay**: learning rate decay over each mini-batch update.
```
Default: --decay 0
```
2. **--momentum**: SGD + momentum
```
Default: --momentum 0
```

# Predict

## Example
```
CUDA_VISIBLE_DEVICES=0 python3 predict.py --bsize 1024 \
						--test_set ./data/mnist-demo.t.mat \
						--model ./saved_model/model.ckpt --dim 28 28 1
```

## Arguments
You may need the following arguments to run the predict script:
1. **--model**: address of the saved model from training
```
Default: --model ./saved_model/model.ckpt
```
2. **--dim**: input dimension of data. Shape must be: height width num_channels
```
Default: --dim 32 32 3
```
3. **--test_set**: provide the directory of .mat file for test.
```
Default: --test_set data/mnist-demo.t.mat
```
4. **--bsize**: Split data into segements of size **bsize** so that each segment can fit into memroy for stochastic gradient and global graident. If you encounter Out of Memory (OOM) during training, you may decrease the **--bsize** paramter to an appropriate value.
```
Default: --bsize 1024
```

# Experiment Results
In the following experiments, we run 100 Newton steps on Newton method and 500 epochs on SGD. We report our resutls on both 4-layer and 7-layer CNN with MSE loss function. We consider the same 4-layer CNN setting in [Wang et al.](https://www.csie.ntu.edu.tw/~cjlin/papers/cnn/newton-CNN.pdf). Other settings such as the initialization are also the same as [Wang et al.](https://www.csie.ntu.edu.tw/~cjlin/papers/cnn/newton-CNN.pdf) for both 4-layer and 7-layer CNN. Both netowrks are trained and tested on CIFAR10 dataset. To reproduce our results, you may download training set [cifar10.mat](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.mat) and test set [cifar10.t.mat](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/cifar10.t.mat) to ./data directory.

## Experiments on 4 layer CNN

### Wall clock time comparison
![](./FILES_FOR_README/ACCU_TIME_3_layers_1.png "accu vs time")

![](./FILES_FOR_README/loss_3layers_1.png "loss of 3 layer CNN")

![](./FILES_FOR_README/accu_3layers_1.png "accuracy of 3 layer CNN")

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

## Experiments on 7 layer CNN
### Wall clock time comparison
![](./FILES_FOR_README/ACCU_TIME_7_layers_1.png "accu vs time")

![](./FILES_FOR_README/loss_7layers_1.png "loss of 6 layer CNN")

![](./FILES_FOR_README/accu_7layers_1.png "accuracy of 6 layer CNN")

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

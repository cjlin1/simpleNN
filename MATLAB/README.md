# Table of contents

- [Requirements](#requirements)
- [_**cnn_train**_ Usage](#_**cnn_train**_-usage)
- [_**cnn_predict**_ Usage](#_**cnn_predict**_-usage)
- [Configuration File](#configuration-file)
- [Options](#options)
- [Data Provided](#data-provided)
- [A Full Example](#a-full-example)
- [Additional Information](#additional-information)

# Requirements

_MATLAB R2016b_ or _Octave 4.0.3_. Other higher version may work as well.

# _**cnn_train**_ Usage

### Syntax

```matlab
model = cnn_train(y, Z, config_file[, options]);
model = cnn_train(y, Z, config_file[, options, seed]);
```

### Parameters

- **y**: a label vector in the range {1, ..., K} for a K-class problem.
- **Z**: a dense feature matrix with the shape of l by a\*b\*d , where l is #instances, a is image height, b is image width and d is #channels.
- **config_file**: a string specifying the configuration file path. Please see the [Configuration File](#configuration-file) section.
- **options**: a string. Please see the [Options](#options) section. If no option, use ''.
- **seed**: a nonnegative integer for _MATLAB_, or an arbitrary vector of length less than or equal to 625 for _Octave_. It is used for the seed of random number generator, _**rng()**_ in _MATLAB_ and _**rand()**_, _**randi()**_ in _Octave_. If it is not given, a different result is produced after each run.

### Returns

- **model**: a structure consists of trained model variables and parameters.

### Example

1. Running with the default options.
```matlab
>> model = cnn_train(y, Z, 'config/mnist-demo-layer3.config');
```

2. Running with the default options and a specific seed.
```matlab
>> model = cnn_train(y, Z, 'config/mnist-demo-layer3.config', '', 111);
```

3. Running with the specific options and no seed.
```matlab
>> model = cnn_train(y, Z, 'config/mnist-demo-layer3.config', '-C 0.01 -SR 0.01');
```

4. Running with the specific options and a specific seed.
```matlab
>> model = cnn_train(y, Z, 'config/mnist-demo-layer3.config', '-C 0.01 -SR 0.01', 111);
```

# _**cnn_predict**_ Usage

### Syntax

```matlab
[predict, acc] = cnn_predict(y, Z, model)
```

### Parameters

- **y**: a label vector in the range {1, ..., K} for a K-class problem.
- **Z**: a dense feature matrix with the shape of l by a\*b\*d, where l is #instances, a is image height, b is image width and d is #channels.
- **model**: a structure consists of trained model variables and parameters.

### Returns

- **predict**: an l by 1 predicted label vector, where l is #instances.
- **acc**: accuracy.

### Example

```matlab
>> [predict, acc] = cnn_predict(y, Z, model);
```

# Configuration File

Let's take _./config/mnist-demo-layer3.config_ as an example. The following items are contained in the configuration file.

1. The #layers of the neural network:
 - **L**: the number of layers
 - **LC**: the number of convolutional layers
 - **LF**: the number of fully-connected layers
```
L = 4;
LC = 3;
LF = 1;
```

2. The statistics of the training data:
 - **wd_input**: the width of the input image data
 - **ht_input**: the height of the input image data
 - **ch_input**: the number of channels in each convolutional layer
```
wd_input = 28;
ht_input = 28;
ch_input = [1,16,16,32];
```

3. The parameters of the neural network structure:
 - **wd_pad_added**: the width of the zero-padding around the input image border at each layer.
 - **wd_filter**: the width of the filters for the convolutional operation at each layer.
 - **strides**: the stride of the filters for the convolutional operation at each layer.
 - **wd_subimage_pool**: the width of the filters for the max pooling operation at each layer. If the pooling operation is not applied for a particular layer, set the corresponding element to 1. For example, if we don't apply the pooling operation at layer 2, the array will be [2,1,2].
 - **full_neurons**: the number of neurons at each fully-connected layer.
```
wd_pad_added = [2,1,1];
wd_filter = [5,3,3];
strides = [1,1,1];
wd_subimage_pool = [2,2,2];
full_neurons = [10];
```

# Options

In this section, we show all the available option/parameters. They are listed along with their default values. You can change the default value by giving the string **option** in _**cnn_train**_. The string format should be

```
'<option> <value>'
```

and separated by a space between each option-value pair inside the string. For example, we can set the **option** to be '-C 0.1 -SR 0.1' to change the regularization constant and the sampling rate for Newton method.

1. **-s**: the optimization method used for training CNN. (1: Newton method (Default); 2: SG method.)
```
solver = 1;
```

### Newton Method

The following options are necessary parameters for Newton method.

1. **-SR**: the sampling rate of the subsampled Gauss-Newton matrix.
```
SR = 0.05;
```

2. **-iter_max**: the maximal number of Newton iterations.
```
iter_max = 100;
```

3. **-C**: the regularization constant in the objective function.
```
C = 0.01;
```

4. **-xi**: the tolerance in the relative stopping condition for the conjugate gradient (CG) method.
```
xi = 0.1;
```

5. **-CGmax**: the maximal number of CG iterations.
```
CGmax = 250;
```

6. **-lambda**: the initial lambda for the Levenberg-Marquardt (LM) method.
```
lambda = 1;
```

7. **-drop**/**-boost**: the drop and boost constants for the LM method.
```
drop = 2/3;
boost = 3/2;
```

8. **-eta**: the parameter for the line search stopping condition.
```
eta = 0.0001;
```

### Stochastic Gradient (SG) Method

The following options are necessary parameters for SG method.

1. **-epoch_max**: the maximal number of SG epochs.
```
epoch_max = 500;
```

2. **-C**: the regularization constant in the objective function.
```
C = 0.01;
```

3. **-lr**: learning rate.
```
lr = 0.01
```

4. **-decay**: learning rate decay over each mini-batch update.
```
decay = 0
```

5. **-bsize**: mini-batch size.
```
bsize = 128
```

6. **-momentum**: weight of information from past sub-gradients.
```
momentum = 0
```

# Data Provided

We provide two small data sets, _mnist-demo.mat_ for training and _mnist-demo.t.mat_ for testing. The data sets are generated using stratified selection to select 2,000 instances and 1,000 instances from [mnist.scale.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2) and [mnist.scale.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2) respectively.

The _*.mat_ files contain a label vector, y and a dense feature matrix, Z. In the original data, the range of the label vector, y, is {0, ..., 9}. We change it to {1, ..., 10} following the requirement of _**cnn_train**_ or _**cnn_predict**_.

For example:

```matlab
>> load('data/mnist-demo.mat', 'y', 'Z');
>> model = cnn_train(y, Z, 'config/mnist-demo-layer3.config');
```


# A Full Example

Please see _./example_mnist.m_.

# Additional Information

For any questions and comments, please email cjlin@csie.ntu.edu.tw

Acknowledgments: this work was supported in part by MOST of Taiwan via the grant 105-2218-E-002-033.


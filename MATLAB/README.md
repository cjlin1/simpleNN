# Table of contents

- [Requirements](#requirements)
- [_**cnn_train**_ Usage](#cnn_train-usage)
- [_**cnn_predict**_ Usage](#cnn_predict-usage)
- [Configuration File](#configuration-file)
- [Data Provided](#data-provided)
- [A Full Example](#a-full-example)
- [Additional Information](#additional-information)

# Requirements

1. _MATLAB R2016b_ or _Octave 4.0.3_. Other higher version may work as well.

2. The GPU version is currently only supported by _MATLAB_. Users need to 
compile simpleNN/MATLAB/cnn/accum.cu first. We take _MATLAB_2017a version as an example.
```
cd simpleNN/MATLAB;
make('/usr/local/cuda-8.0/bin');
```
For the cuda version suppored by _MATLAB_ release, please check the following website:
https://www.mathworks.com/help/parallel-computing/gpu-support-by-release.html.

# _**cnn_train**_ Usage

### Syntax

```matlab
model = cnn_train(y, Z, [], [], config_file[, options]);
model = cnn_train(y, Z, [], [], config_file[, options, seed]);
model = cnn_train(y, Z, y_v, Z_v, config_file[, options]);
model = cnn_train(y, Z, y_v, Z_v, config_file[, options, seed]);
```
### Parameters

- **y** and **y_v**: a vector of integers to indicate class labels.
- **Z** and **Z_v**: a dense feature matrix with the shape of l by a\*b\*d , where l is #instances, a is image height, b is image width and d is #channels.
  - (**y**, **Z**) represents a training data and (**y_v**, **Z_v**) represents a validation data.
- **config_file**: a string specifying the configuration file path. Please see the [Configuration File](#configuration-file) section.
- **options**: a string. Please see the [Options](#options) part in this section. If no option, use ''.
- **seed**: a nonnegative integer for _MATLAB_, or an arbitrary vector of length less than or equal to 625 for _Octave_. It is used for the seed of random number generator, _**rng()**_ in _MATLAB_ and _**rand()**_, _**randi()**_ in _Octave_. If it is not given, a different result is produced after each run.

### Returns

- **model**: a structure consists of trained model variables and parameters. If (**y_v**, **Z_v**) is provided, 
we select the output **model** with the best validation accuracy. 
If not, we select the model of the last iteration to be the output **model**.  

### Example

1. Running with the default options.
```matlab
>> model = cnn_train(y, Z, [], [], 'config/mnist-demo-layer3.config');
```

2. Running with the default options and a specific seed.
```matlab
>> model = cnn_train(y, Z, [], [], 'config/mnist-demo-layer3.config', '', 111);
```

3. Running with the specific options and no seed.
```matlab
>> model = cnn_train(y, Z, [], [], 'config/mnist-demo-layer3.config', '-C 0.01 -GNsize 100');
```

4. Running with the specific options and a specific seed.
```matlab
>> model = cnn_train(y, Z, [], [], 'config/mnist-demo-layer3.config', '-C 0.01 -GNsize 100', 111);
```

### Options

In this section, we show all the available option/parameters. They are listed along with their default values. You can change the default value by giving the string **option** in _**cnn_train**_. The string format should be

```
'<option> <value>'
```

and separated by a space between each option-value pair inside the string. For example, we can set the **option** to be '-C 0.1 -GNsize 128' to change the regularization constant and the number of instances selected from the data for Newton method.

1. **-s**: the optimization method used for training CNN. (1: Newton method (Default); 2: SG method.)
```
solver = 1;
```

2. **-C**: the regularization constant in the objective function.
```
C = 0.01;
```

3. **-gpu_use**: using GPU or not (0: GPU not used; 1: GPU used). Default 1 if GPU devices detected; 0 otherwise.

4. **-ftype**: the precision of the floating point (0: single; 1: double). Default: 0 if gpu_use = 1; 1 otherwise.

5. **-bsize**: mini-batch size. For SG, the number of data per update. For Newton, it is the batch size in function and gradient evaluations, 
and subsampled Gauss-Newton matrix-vector products. Default: 1024 for Newton without using GPU; 128 otherwise.

#### Newton Method

The following options are necessary parameters for Newton method.

1. **-GNsize**: the number of instances selected from the data for the subsampled Gauss-Newton matrix.
```
GNsize = ceil(0.05*total_number_instances);
```

2. **-iter_max**: the maximal number of Newton iterations.
```
iter_max = 100;
```

3. **-xi**: the tolerance in the relative stopping condition for the conjugate gradient (CG) method.
```
xi = 0.1;
```

4. **-CGmax**: the maximal number of CG iterations.
```
CGmax = 250;
```

5. **-lambda**: the initial lambda for the Levenberg-Marquardt (LM) method.
```
lambda = 1;
```

6. **-drop**/**-boost**: the drop and boost constants for the LM method.
```
drop = 2/3;
boost = 3/2;
```

7. **-eta**: the parameter for the line search stopping condition.
```
eta = 0.0001;
```

8. **-Jacobian**: the flag of storing the Jacobian matrix.
```
if gpu_use
    param.Jacobian = false;
else
    param.Jacobian = true;
end
```

#### Stochastic Gradient (SG) Method

The following options are necessary parameters for SG method.

1. **-epoch_max**: the maximal number of SG epochs.
```
epoch_max = 500;
```

2. **-lr**: learning rate.
```
lr = 0.01
```

3. **-decay**: learning rate decay over each mini-batch update.
```
decay = 0
```

4. **-momentum**: weight of information from past sub-gradients.
```
momentum = 0
```

# _**cnn_predict**_ Usage

### Syntax

1. Running with the default options.
```matlab
>> [predict, acc] = cnn_predict(y, Z, model)
```

2. Running with the specific options.
```matlab
>> [predict, acc] = cnn_predict(y, Z, model, '-bsize 64')
```

### Parameters

- **y**: a vector of integers to indicate class labels.
- **Z**: a dense feature matrix with the shape of l by a\*b\*d, where l is #instances, a is image height, b is image width and d is #channels.
- **model**: a structure consists of trained model variables and parameters.
- **options**: a string. Please see the [Options](#options-1) part in this section. If no option, use ''.

### Returns

- **predict**: an l by 1 predicted label vector, where l is #instances.
- **acc**: accuracy.

### Example

```matlab
>> [predict, acc] = cnn_predict(y, Z, model);
```
### Options

In this subsection, we show all the available options/parameters. They are listed along with their default values. You can change the default value by giving the string **option** in _**cnn_predict**_ and the string format is the same as 
the option string format for _**cnn_train**_.

1. **-bsize**: mini-batch size.
```
bsize = 128;
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

# Data Provided

We provide two small data sets, _mnist-demo.mat_ for training and _mnist-demo.t.mat_ for testing. The data sets are generated using stratified selection to select 2,000 instances and 1,000 instances from [mnist.scale.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2) and [mnist.scale.t.bz2](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2) respectively.

The _*.mat_ files contain a label vector, y and a dense feature matrix, Z.

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


function example(options, input_format)

if nargin == 0
	options = '';
end
if nargin <= 1
	input_format = 0;
end
if (input_format ~= 0) && (input_format ~= 1)
	error('input_format must be 0 or 1.');
end

% To access read_config.m file in cnn directory
addpath(genpath('./cnn'));

%% Train
% ------
config_file = 'config/mnist-demo-layer3.config';
net_config = read_config(config_file);
a = net_config.ht_input(1);
b = net_config.wd_input(1);
d = net_config.ch_input(1);

% Read train data sets
load('data/mnist-demo.mat', 'y', 'Z');
% Because sparse matrices stored in the provided mat file do not store zero columns in the end, we need to fill it.
Z = [full(Z) zeros(size(Z,1), a*b*d - size(Z,2))];

% If input data format is row-wise, we rearrange data from row-wise to col-wise
if input_format == 0
	Z = reshape(permute(reshape(Z, [],b,a,d), [1,3,2,4]), [], a*b*d);
end

% Normalization
Z = Z / 255;

% Zero mean
mean_tr = mean(Z);
Z = Z - mean_tr;

seed = 111;

model = cnn_train(y, Z, [], [], config_file, options, seed);

%% Test
% -----
% Read test data sets
load('data/mnist-demo.t.mat', 'y', 'Z');
% Because sparse matrices stored in the provided mat file do not store zero columns in the end, we need to fill it.
Z = [full(Z) zeros(size(Z,1), a*b*d - size(Z,2))];

% If input data format is row-wise, we rearrange data from row-wise to col-wise
if input_format == 0
	Z = reshape(permute(reshape(Z, [],b,a,d), [1,3,2,4]), [], a*b*d);
end

% Normalization
Z = Z / 255;

% Zero mean
Z = Z - mean_tr;

[predicted_label, acc] = cnn_predict(y, Z, model);
fprintf('test_acc: %g\n', acc);


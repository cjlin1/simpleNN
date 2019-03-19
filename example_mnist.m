function example_mnist(options)

if nargin == 0
	options = '';
end

%% Train
% ------
config_file = 'config/mnist-demo-layer3.config';

% Read train data sets
load('data/mnist-demo.mat', 'y', 'Z');

% Max-min normalization
tmp_max = max(Z, [], 2);
tmp_min = min(Z, [], 2);
Z = (Z - tmp_min) ./ (tmp_max - tmp_min);

% Zero mean
mean_tr = mean(Z);
Z = Z - mean_tr;

model = cnn_train(y, Z, config_file, options, 111);

%% Test
% -----
% Read test data sets
load('data/mnist-demo.t.mat', 'y', 'Z');

% Max-min normalization
tmp_max = max(Z, [], 2);
tmp_min = min(Z, [], 2);
Z = (Z - tmp_min) ./ (tmp_max - tmp_min);

% Zero mean
Z = Z - mean_tr;

[predicted_label, acc] = cnn_predict(y, Z, model);
fprintf('test_acc: %g\n', acc);


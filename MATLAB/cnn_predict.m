function [results, acc] = cnn_predict(y, Z, model, options)

if nargin == 3
	options = '';
end

addpath(genpath('./cnn'), genpath('./opt'));

% Inherit gpu_use and float_type properties from the training procedure
global gpu_use;
gpu_use = model.gpu_use;

global float_type;
float_type = model.float_type;

param = parameter(options);
prob = check_data(y, Z, model.net_config);
net = init_net(model, param.bsize);

results = predict(prob, param, model, net);

% Obtain predicted label
[~, results] = max(results, [], 1);
results = results';

acc = cal_accuracy(results, y);

function param = parameter(options)

param = struct;
param.bsize = 128;

% Read options given by users
if ~isempty(options)
	param = parse_options(param, options);
end

function param = parse_options(param, options)

options = strsplit(strtrim(options), ' ');

if mod(length(options), 2) == 1
	error('Each option is specified by its name and value.');
end

for i = 1 : length(options)/2
	option = options{2*i-1};
	value = str2num(options{2*i});
	switch option
		case '-bsize'
			param.bsize = value;
		otherwise
			error('%s is not a supported option.', option);
	end
end

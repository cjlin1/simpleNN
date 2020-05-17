function model = cnn_train(y, Z, y_v, Z_v, config_file, options, seed)

if nargin == 5 || nargin == 6
	if nargin == 5
		options = '';
	end
	if exist('OCTAVE_VERSION', 'builtin')
		rand('state');
		randn('state');
	else
		rng('shuffle');
	end
elseif nargin == 7
	if exist('OCTAVE_VERSION', 'builtin')
		rand('state', seed);
		randn('state', seed);
	else
		rng(seed);
	end
else
	error('The #arguments is incorrect.');
end
addpath(genpath('./cnn'), genpath('./opt'));

param = parameter(Z, options);
net_config = read_config(config_file);
prob = check_data(y, Z, net_config);
% check if the vaildation data exists
prob_v = struct;
if ~isempty(y_v) && ~isempty(Z_v)
	prob_v = check_data(y_v, Z_v, net_config);
end

model = train(prob, prob_v, param, net_config);

function param = parameter(Z, options)

param = struct;

param.solver = 1;
param.C = 0.01;
param.bsize = [];

% parameters for Newton methods

% The subsampled size for calculating the Gauss-Newton matrix
param.GNsize = ceil(0.05*size(Z,1));
% The maximum number of Newton iterations
param.iter_max = 100;
% CG
param.xi = 0.1;
param.CGmax = 250;
% Levenberg-Marquardt method
param.lambda = 1;
param.drop = 2/3;
param.boost = 3/2;
% line search
param.eta = 1e-4;

% Check GPU device
global gpu_use;
if exist('OCTAVE_VERSION', 'builtin')
	gpu_use = false;
else
	gpu_use = (gpuDeviceCount > 0);
end

% floating-point type and the flag of storing Jacobian
global float_type;
float_type = [];
param.Jacobian = [];

% parameters for stochastic gradient

param.epoch_max = 500;
param.lr = 0.01;
param.decay = 0;
param.momentum = 0;

% Read options given by users
if ~isempty(options)
	param = parse_options(param, options);
end

if isempty(float_type)
    if gpu_use
        float_type = 'single';
    else
        float_type = 'double';
    end
end

if isempty(param.Jacobian)
    param.Jacobian = ~gpu_use;
end

if isempty(param.bsize)
	param.bsize = 128;
	if ~gpu_use && (param.solver == 1)
		param.bsize = 1024;
	end
end

param.C = param.C*size(Z,1);

function param = parse_options(param, options)

options = strsplit(strtrim(options), ' ');

if mod(length(options), 2) == 1
	error('Each option is specified by its name and value.');
end

global gpu_use float_type;
for i = 1 : length(options)/2
	option = options{2*i-1};
	value = str2num(options{2*i});
	switch option
		case '-s'
			param.solver = value;
		case '-GNsize'
			param.GNsize = value;
		case '-iter_max'
			param.iter_max = value;
		case '-C'
			param.C = value;
		case '-xi'
			param.xi = value;
		case '-CGmax'
			param.CGmax = value;
		case '-lambda'
			param.lambda = value;
		case '-drop'
			param.drop = value;
		case '-boost'
			param.boost = value;
		case '-eta'
			param.eta = value;
		case '-lr'
			param.lr = value;
		case '-decay'
			param.decay = value;
		case '-bsize'
			param.bsize = value;
		case '-momentum'
			param.momentum = value;
		case '-epoch_max'
			param.epoch_max = value;
		case '-Jacobian'
			param.Jacobian = logical(value);
		case '-ftype'
			if value == 0
				float_type = 'single';
			elseif value == 1
				float_type = 'double';
			else
				error('we do not support this float type');
			end
		case '-gpu_use'
			if ~(gpu_use) && logical(value)
				error('we do not detect any gpuDevice.');
			end
			gpu_use = logical(value) && (exist('OCTAVE_VERSION', 'builtin') == 0);
		otherwise
			error('%s is not a supported option.', option);
	end
end


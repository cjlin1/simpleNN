function model = cnn_train(y, Z, config_file, options, seed)
% -s: the optimization method used for training CNN. (1: Newton method (Default); 2: SG method)
% -C: the regularization constant in the objective function.
% -inner_bsize: the smaller inner batch size than a mini-batch or a subsampled subset.
% -gpu_use the flag (0: CPU; 1: GPU. default:0)
%
% options for Newton method (-s 1):
% -SR: the sampling rate of the subsampled Gauss-Newton matrix.
% -iter_max: the maximal number of Newton iterations.
% -xi: the tolerance in the relative stopping condition for the conjugate gradient (CG) method.
% -CGmax: the maximal number of CG iterations.
% -lambda: the initial lambda for the Levenberg-Marquardt (LM) method.
% -drop/-boost: the drop and boost constants for the LM method.
% -eta: the parameter for the line search stopping condition.
% -JF: Jacobian free strategy. (0: store dZdS. 1: calculate dZdS every CG iteration. default: 0)
%
% options for SG method (-s 2):
% -epoch_max: the maximal number of SG epochs.
% -lr: learning rate.
% -bsize: mini-batch size.
% -momentum: weight of information from past sub-gradients.
% -decay: learning rate decay over each mini-batch update.
%

if nargin == 3 || nargin == 4
	if nargin == 3
		options = '';
	end
	if exist('OCTAVE_VERSION', 'builtin')
		rand('state');
		randn('state');
	else
		rng('shuffle');
	end
elseif nargin == 5
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

% Check GPU device
global gpu_use
gpu_use = (gpuDeviceCount > 0);

param = parameter(y, Z, config_file, options);
if (gpu_use == 0)
	fprintf('We use CPUs to train\n');
else
	fprintf('We use GPUs to train\n');
end
prob = check_data(y, Z, param);
model = train(prob, param);

function param = parameter(y, Z, config_file, options)

param = struct;

param.solver = 1;
param.inner_bsize = 128;

% parameters for Newton methods
param.JF = 0;

% The subsampled size for calculating the Gauss-Newton matrix
param.SR = 0.05;
% The maximum number of Newton iterations
param.iter_max = 100;
% Objective function
param.C = 0.01;
% CG
param.xi = 0.1;
param.CGmax = 250;
% Levenberg-Marquardt method
param.lambda = 1;
param.drop = 2/3;
param.boost = 3/2;
% line search
param.eta = 1e-4;

% parameters for stochastic gradient

param.lr = 0.01;
param.decay = 0;
param.bsize = 128;
param.momentum = 0;
param.epoch_max = 500;

% Read options given by users
if ~isempty(options)
	param = parse_options(param, options);
end

param.C = param.C*size(Z,1);
param.num_splits = floor(1/param.SR);

% Read Config
fid = fopen(config_file, 'r');
if fid == -1
	error('The configure file cannot be opened.');
end
while ~feof(fid)
	s = fgetl(fid);
	if ~isempty(s)
		if strcmp(s(1),'%') == 0
			eval(['param.' s]);
		end
	end
end
fclose(fid);

if param.LC == 0
	error('You must have at least one convolutional layer.');
end

param.nL = param.full_neurons(param.LF);

function param = parse_options(param, options)

options = strsplit(strtrim(options), ' ');

if mod(length(options), 2) == 1
	error('Each option is specified by its name and value.');
end
global gpu_use
for i = 1 : length(options)/2
	option = options{2*i-1};
	value = str2num(options{2*i});
	switch option
		case '-s'
			param.solver = value;
		case '-SR'
			param.SR = value;
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
		case '-inner_bsize'
			param.inner_bsize = value;
		case '-JF'
			param.JF = value;
		case '-gpu_use'
			if (gpu_use == 0 & value == 1)
				fprintf('[Warning] We do not detect any GPU device.\n');
			end
			if (gpu_use == 1 & value == 0)
				fprintf('[Warning] We detect your GPU device. However, CPUs are chosen to train.\n');
				gpu_use = 0;
			end
		otherwise
			error('%s is not a supported option.', option);
	end
end

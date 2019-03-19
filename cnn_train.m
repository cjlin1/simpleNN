function model = cnn_train(y, Z, config_file, options, seed)

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

param = parameter(y, Z, config_file, options);
prob = check_data(y, Z, param);
model = train(prob, param);

function param = parameter(y, Z, config_file, options)

param = struct;

param.solver = 1;

% parameters for Newton methods

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
param.bsize = 128;

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
		case '-bsize'
			param.bsize = value;
		otherwise
			error('%s is not a supported option.', option);
	end
end


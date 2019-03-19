function model = train(prob, param)

model = init_model(param);
net = init_net(param, model);

switch param.solver
	case 1
		model = newton(prob, param, model, net);
	case 2
		model = sgd(prob, param, model, net);
	case 3
		model = adam(prob, param, model, net);
otherwise
	error('solver not correctly specified', param.solver);
end

function model = init_model(param)

LC = param.LC;
L = param.L;
model = struct;
model.ht_input = [param.ht_input; zeros(LC, 1)];  % height of input image
model.wd_input = [param.wd_input; zeros(LC, 1)];  % width of input image
model.ch_input = param.ch_input(:);  % #channels of input image
model.wd_pad_added = param.wd_pad_added(:);  % width of zero-padding around input image border
model.ht_pad = [zeros(LC, 1)];  % height of image after padding
model.wd_pad = [zeros(LC, 1)];  % width of image after padding
model.ht_conv = [zeros(LC, 1)];  % height of image after convolution
model.wd_conv = [zeros(LC, 1)];  % width of image after convolution
model.wd_filter = param.wd_filter(:);  % width of filter in convolution
model.strides = param.strides(:);  % strides of convolution
model.wd_subimage_pool = param.wd_subimage_pool(:);  % width of filter in pooling
model.full_neurons = param.full_neurons(:);  % #neurons in fully-connected layers
model.weight = cell(L, 1);
model.bias = cell(L, 1);
var_num = zeros(L, 1);

for m = 1 : LC
	model.ht_pad(m) = model.ht_input(m) + 2*model.wd_pad_added(m);
	model.wd_pad(m) = model.wd_input(m) + 2*model.wd_pad_added(m);
	model.ht_conv(m) = floor((model.ht_pad(m) - model.wd_filter(m))/model.strides(m)) + 1;
	model.wd_conv(m) = floor((model.wd_pad(m) - model.wd_filter(m))/model.strides(m)) + 1;
	model.ht_input(m+1) = floor(model.ht_conv(m)/model.wd_subimage_pool(m));
	model.wd_input(m+1) = floor(model.wd_conv(m)/model.wd_subimage_pool(m));
	var_num(m) = model.ch_input(m+1)*(model.wd_filter(m)*model.wd_filter(m)*model.ch_input(m) + 1);

	model.weight{m} = randn(model.ch_input(m+1),model.wd_filter(m)*model.wd_filter(m)*model.ch_input(m))*sqrt(2.0/(model.wd_filter(m)*model.wd_filter(m)*model.ch_input(m)));
	model.bias{m} = zeros(model.ch_input(m+1),1);
end

num_neurons_prev = model.ht_input(LC+1)*model.wd_input(LC+1)*model.ch_input(LC+1);
for m = LC+1 : L
	num_neurons = model.full_neurons(m - LC);
	model.weight{m} = randn(num_neurons, num_neurons_prev) * sqrt(2.0/num_neurons_prev);
	model.bias{m} = zeros(num_neurons, 1);
	var_num(m) = num_neurons * (num_neurons_prev + 1);
	num_neurons_prev = num_neurons;
end
% starting index of trained variables (including biases) for each layer
model.var_ptr = [1; cumsum(var_num)+1];

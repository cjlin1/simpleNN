function prob = check_data(y, Z, param)

% Check class labels
label_enum = sort(unique(y))';
if length(label_enum) ~= param.nL
	error('One or more training labels are missing.');
end
if ~all(label_enum == 1 : param.nL)
	error('Training labels should be the same as {1, ..., #classes}.');
end

% Construct one hot label
num_class = param.nL;
l = length(y);
prob.label_mat = zeros(num_class, l);
prob.label_mat(y + num_class*[0:l-1]') = 1;

if issparse(Z)
	error('The feature matrix must be dense.');
end

a = param.ht_input;
b = param.wd_input;

if size(Z, 2) ~= param.ch_input(1)*a*b
	error('The #columns in the feature matrix must be equal to %d*%d*%d (wd_input*ht_input*ch_input(1)).', b, a, model.ch_input(1));
end

% Initialize #instances
prob.l = size(Z, 1);

% Rearrange data from the shape of l x abd to the shape of dab x l
tmp = [];
for d = 1 : param.ch_input(1)
	tmp = [tmp; reshape(Z(:, (d-1)*a*b+1 : d*a*b)',[],1)'];
end
prob.data = reshape(tmp, [], prob.l);

function prob = check_data(y, Z, net_config)

% Check class labels
if length(unique(y)) ~= net_config.nL
	error('# labels in data different from # last-layer nodes specified in configuration.');
end

prob.y = y;

if issparse(Z)
	error('The feature matrix must be dense.');
end

a = net_config.ht_input;
b = net_config.wd_input;

if size(Z, 2) ~= net_config.ch_input(1)*a*b
	error('The #columns in the feature matrix must be equal to %d*%d*%d (wd_input*ht_input*ch_input(1)).', b, a, model.ch_input(1));
end

% Initialize #instances
prob.l = size(Z, 1);

% Rearrange data from the shape of l x abd to the shape of dab x l
tmp = [];
for d = 1 : net_config.ch_input(1)
	tmp = [tmp; reshape(Z(:, (d-1)*a*b+1 : d*a*b)', [], 1)'];
end
prob.data = reshape(tmp, [], prob.l);

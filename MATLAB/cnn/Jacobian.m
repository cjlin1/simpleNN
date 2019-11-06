function net = Jacobian(data, param, model, net)

L = model.L;
num_data = size(data, 2);

bsize = param.bsize;
num_batches = ceil(num_data/bsize);
net.dzdS = cell(L*num_batches, 1);

for i = 1 : num_batches
	range = (i-1)*bsize + 1 : min(num_data, i*bsize);

	% net.Z
	net = feedforward(data(:, range), model, net);

	% Compute dzdS
	dzdS = cal_dzdS(data(:, range), model, net);

	% store dzdS
	for m = 1 : L
		net.dzdS{(i-1)*L + m} = dzdS{m};
	end
end

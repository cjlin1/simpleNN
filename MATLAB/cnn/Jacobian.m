function net = Jacobian(data, param, model, net)

L = model.L;
LC = model.LC;
num_data = size(data, 2);

bsize = param.bsize;
num_batches = ceil(num_data/bsize);
net.dzdS = cell(L*num_batches, 1);
J_Z = cell(L*num_batches, 1);
J_phiZ = cell(L*num_batches, 1);

for i = 1 : num_batches
	range = (i-1)*bsize + 1 : min(num_data, i*bsize);

	% net.Z
	net = feedforward(data(:, range), model, net, 'not_Jv');

	% Compute dzdS
	dzdS = cal_dzdS(model, net, length(range));

	% store Z, phiZ, and dzdS
	for m = 1 : L
		if m > LC
			J_Z{(i-1)*L + m} = net.Z{m};
		else
			J_phiZ{(i-1)*L + m} = net.phiZ{m};
		end
		net.dzdS{(i-1)*L + m} = dzdS{m};
	end
end

net.Z = J_Z;
net.phiZ = J_phiZ;

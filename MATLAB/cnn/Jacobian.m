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

function dzdS = cal_dzdS(model, net, num_data)

L = model.L;
LC = model.LC;
nL = model.nL; 
dzdS = cell(L, 1);

dzdS{L} = repmat(gpu(@eye, [nL, nL]), 1, num_data);

for m = L : -1 : max(LC+1, 2)
	dzdS{m-1} = (model.weight{m}' * dzdS{m}).*reshape(repmat(net.Z{m} > 0, nL, 1), [], nL*num_data);
end

for m = LC : -1 : 1
	if model.wd_subimage_pool(m) > 1 
		dzdS{m} = vTP(model, net, m, num_data, dzdS{m}, 'pool_Jacobian');
	end

	dzdS{m} = reshape(dzdS{m}, model.ch_input(m+1), []);

	if m > 1
		V = model.weight{m}' * dzdS{m};
		dzdS{m-1} = vTP(model, net, m, num_data, V, 'phi_Jacobian');

		% vTP_pad 
		dzdS{m-1} = reshape(dzdS{m-1}, model.ch_input(m), model.ht_pad(m), model.wd_pad(m), []);
		p = model.wd_pad_added(m);
		dzdS{m-1} = dzdS{m-1}(:, p+1:p+model.ht_input(m), p+1:p+model.wd_input(m), :);

		dzdS{m-1} = reshape(dzdS{m-1}, [], nL, num_data) .* reshape(net.Z{m} > 0, [], 1, num_data);
	end
end

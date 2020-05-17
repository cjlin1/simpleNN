function net = Jacobian(data, param, model, net)

% net.Z
net = feedforward(data, model, net);

% net.dzdS
net = cal_dzdS(model, net, size(data, 2));

function net = cal_dzdS(model, net, num_data)

L = model.L;
LC = model.LC;
nL = model.nL; 
net.dzdS = cell(L, 1);

net.dzdS{L} = repmat(gpu(@eye, [nL, nL]), 1, num_data);

for m = L : -1 : max(LC+1, 2)
	net.dzdS{m-1} = (model.weight{m}' * net.dzdS{m}).*reshape(repmat(net.Z{m} > 0, nL, 1), [], nL*num_data);
end

for m = LC : -1 : 1
	if model.wd_subimage_pool(m) > 1 
		net.dzdS{m} = vTP(model, net, m, num_data, net.dzdS{m}, 'pool_Jacobian');
	end

	net.dzdS{m} = reshape(net.dzdS{m}, model.ch_input(m+1), []);

	if m > 1
		v = model.weight{m}' * net.dzdS{m};
		net.dzdS{m-1} = vTP(model, net, m, num_data, v, 'phi_Jacobian');

		% vTP_pad 
		net.dzdS{m-1} = reshape(net.dzdS{m-1}, model.ch_input(m), model.ht_pad(m), model.wd_pad(m), []);
		p = model.wd_pad_added(m);
		net.dzdS{m-1} = net.dzdS{m-1}(:, p+1:p+model.ht_input(m), p+1:p+model.wd_input(m), :);

		net.dzdS{m-1} = reshape(net.dzdS{m-1}, [], nL, num_data) .* reshape(net.Z{m} > 0, [], 1, num_data);
	end
end

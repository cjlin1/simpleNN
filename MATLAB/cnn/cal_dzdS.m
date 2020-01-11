function dzdS = cal_dzdS(data, model, net)

L = model.L;
LC = model.LC;
nL = model.nL;
dzdS = cell(L, 1);
num_data = size(data, 2);

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


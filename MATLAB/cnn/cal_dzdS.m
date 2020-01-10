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
		dzdS{m-1} = reshape(vTP(model, net, m, num_data, V, 'phi_Jacobian'), model.ch_input(m), []);

		% vTP_pad
		a = model.ht_pad(m); b = model.wd_pad(m);
		dzdS{m-1} = dzdS{m-1}(:, net.idx_pad{m} + a*b*[0:nL*num_data-1]);
		dzdS{m-1} = reshape(dzdS{m-1}, [], nL, num_data) .* reshape(net.Z{m} > 0, [], 1, num_data);
	end
end


function JTv_ = JTv(model, net, dXidS)

L = model.L;
LC = model.LC;
var_ptr = model.var_ptr;
num_data = size(dXidS, 2);
JTv_ = cell(L, 1);

for m = L : -1 : LC+1
	JTv_{m} = [dXidS*net.Z{m}' sum(dXidS, 2)];
	dXidS = model.weight{m}'*dXidS;
	dXidS = dXidS.*(net.Z{m} > 0);
end
dXidS = reshape(dXidS, model.ch_input(LC+1), []);

for m = LC : -1 : 1
	if model.wd_subimage_pool(m) > 1
		dXidS = vTP(model, net, m, num_data, dXidS, 'pool_gradient');
	end
	dXidS = reshape(dXidS, model.ch_input(m+1), []);

	JTv_{m} = [dXidS*net.phiZ{m}' sum(dXidS, 2)];

	if m > 1
		V = model.weight{m}' * dXidS;
		dXidS = vTP(model, net, m, num_data, V, 'phi_gradient');

		% vTP_pad
		dXidS = reshape(dXidS, model.ch_input(m), model.ht_pad(m), model.wd_pad(m), []);
		p = model.wd_pad_added(m);
		dXidS = dXidS(:, p+1:p+model.ht_input(m), p+1:p+model.wd_input(m), :);

		% activation function
		dXidS = reshape(dXidS, model.ch_input(m), []) .*(net.Z{m} > 0);
	end
end

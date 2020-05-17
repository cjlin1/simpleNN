function JTv_ = JTv(model, net, v)

L = model.L;
LC = model.LC;
num_data = size(v, 2);
JTv_ = cell(L, 1);

for m = L : -1 : LC+1
	JTv_{m} = [v*net.Z{m}' sum(v, 2)];
	v = model.weight{m}'*v;
	v = v.*(net.Z{m} > 0);
end
v = reshape(v, model.ch_input(LC+1), []);

for m = LC : -1 : 1
	if model.wd_subimage_pool(m) > 1
		v = vTP(model, net, m, num_data, v, 'pool_gradient');
	end
	v = reshape(v, model.ch_input(m+1), []);

	JTv_{m} = [v*net.phiZ{m}' sum(v, 2)];

	if m > 1
		v = model.weight{m}' * v;
		v = vTP(model, net, m, num_data, v, 'phi_gradient');

		% vTP_pad
		v = reshape(v, model.ch_input(m), model.ht_pad(m), model.wd_pad(m), []);
		p = model.wd_pad_added(m);
		v = v(:, p+1:p+model.ht_input(m), p+1:p+model.wd_input(m), :);

		% activation function
		v = reshape(v, model.ch_input(m), []) .*(net.Z{m} > 0);
	end
end

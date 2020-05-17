function [R_Z, net] = Jv(data, model, net, v)

L = model.L;
LC = model.LC;
var_ptr = model.var_ptr;
num_data = size(data, 2);

net.Z{1} = reshape(gpu(data), model.ch_input(1), []);
R_Z = gpu(@zeros, size(net.Z{1}));

for m = 1 : LC
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	d = model.ch_input(m+1);
	v_ = reshape(v(var_range), d, []);

	net.phiZ{m} = padding_and_phiZ(model, net, net.Z{m}, m, num_data);
	net.Z{m+1} = max(model.weight{m}*net.phiZ{m} + model.bias{m}, 0);

	R_Z = padding_and_phiZ(model, net, R_Z, m, num_data);
	R_Z = model.weight{m}*R_Z + v_(:, 1:end-1)*net.phiZ{m} + v_(:, end);

	if model.wd_subimage_pool(m) > 1
		[net.Z{m+1}, net.idx_pool{m}, net.R_max_id{m}] = maxpooling(model, net, net.Z{m+1}, m, 'Jv');
		R_Z = maxpooling(model, net, R_Z, m, 'R');
	end

	if m == LC
		net.Z{m+1} = reshape(net.Z{m+1}, [], num_data);
		R_Z = reshape(R_Z, [], num_data);
	end
	R_Z = (net.Z{m+1} > 0).*R_Z;
end

for m = LC+1 : L
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	n_m = model.full_neurons(m-LC);
	v_ = reshape(v(var_range), n_m, []);

	R_Z = model.weight{m}*R_Z + v_(:, 1:end-1)*net.Z{m} + v_(:, end);
	if m < L
		net.Z{m+1} = max(model.weight{m}*net.Z{m} + model.bias{m}, 0);
		R_Z = (net.Z{m+1} > 0).*R_Z;
	end
end

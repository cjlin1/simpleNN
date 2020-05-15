function [Jv_, net] = Jv(data, model, net, v_w, v_b)

L = model.L;
LC = model.LC;
num_data = size(data, 2);

% R_feedforward
net.Z{1} = reshape(gpu(data), model.ch_input(1), []);
R_Z = gpu(@zeros, size(net.Z{1}));

for m = 1 : LC
	net.phiZ{m} = padding_and_phiZ(model, net, net.Z{m}, m, num_data);
	net.Z{m+1} = max(model.weight{m}*net.phiZ{m} + model.bias{m}, 0);

	R_Z = padding_and_phiZ(model, net, R_Z, m, num_data);
	R_Z = model.weight{m}*R_Z + v_w{m}*net.phiZ{m} + v_b{m};

	if model.wd_subimage_pool(m) > 1
		[net.Z{m+1}, net.idx_pool{m}, net.R_max_id{m}] = maxpooling(model, net, net.Z{m+1}, m, 'Jv_maxpooling');
		R_Z = maxpooling(model, net, R_Z, m, 'R_maxpooling');
	end

	if m == LC
		net.Z{m+1} = reshape(net.Z{m+1}, [], num_data);
		R_Z = reshape(R_Z, [], num_data);
	end
	R_Z = (net.Z{m+1} > 0).*R_Z;
end

for m = LC+1 : L-1
	R_Z = model.weight{m}*R_Z + v_w{m}*net.Z{m} + v_b{m};
	net.Z{m+1} = max(model.weight{m}*net.Z{m} + model.bias{m}, 0);
	R_Z = (net.Z{m+1} > 0).*R_Z;
end

Jv_ = model.weight{L}*R_Z + v_w{L}*net.Z{L} + v_b{L};

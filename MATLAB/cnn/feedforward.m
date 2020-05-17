function net = feedforward(data, model, net)

num_data = size(data, 2);
net.Z{1} = reshape(gpu(data), model.ch_input(1), []);

L = model.L;
LC = model.LC;

for m = 1 : LC
	net.phiZ{m} = padding_and_phiZ(model, net, net.Z{m}, m, num_data);
	net.Z{m+1} = max(model.weight{m}*net.phiZ{m} + model.bias{m}, 0);

	if model.wd_subimage_pool(m) > 1
		[net.Z{m+1}, net.idx_pool{m}] = maxpooling(model, net, net.Z{m+1}, m);
	end
end

dab = model.ch_input(LC+1) * model.wd_input(LC+1) * model.ht_input(LC+1);
net.Z{LC+1} = reshape(net.Z{LC+1}, dab, []);

for m = LC+1 : L-1
	net.Z{m+1} = max(model.weight{m}*net.Z{m} + model.bias{m}, 0);
end

net.Z{L+1} = model.weight{L}*net.Z{L} + model.bias{L};

function [Jv_, net] = Jv(data, model, net, v_w, v_b)

L = model.L;
LC = model.LC;
num_data = size(data, 2);

% feeforward
net = feedforward(data, model, net, 'Jv');

% R_feedforward
R_Z = gpu(@zeros, size(net.Z{1}));

for m = 1 : LC
    R_Z = padding_and_phiZ(model, net, R_Z, m, num_data);
	R_Z = model.weight{m}*R_Z + v_w{m}*net.phiZ{m} + v_b{m};

    if model.wd_subimage_pool(m) > 1
        R_Z = maxpooling(model, net, R_Z, m, 'R_maxpooling');
    end

	if m == LC
		R_Z = reshape(R_Z, [], num_data);
	end
    R_Z = (net.Z{m+1} > 0).*R_Z;
end

for m = LC+1 : L-1
    R_Z = model.weight{m}*R_Z + v_w{m}*net.Z{m} + v_b{m};
    R_Z = (net.Z{m+1} > 0).*R_Z;
end

Jv_ = model.weight{L}*R_Z + v_w{L}*net.Z{L} + v_b{L};

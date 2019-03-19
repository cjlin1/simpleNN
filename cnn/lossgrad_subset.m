function [net, loss] = lossgrad_subset(prob, param, model, net, batch_idx, task)

L = param.L;
LC = param.LC;

net.num_sampled_data = length(batch_idx);
num_data = net.num_sampled_data;

Y = prob.label_mat(:, batch_idx);
net.Z{1} = reshape(prob.data(:, batch_idx), model.ch_input(1), []);

% fun
net = feedforward(prob.data(:, batch_idx), param, model, net);
loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
	dXidS = 2*(net.Z{L+1} - Y);
	for m = L : -1 : LC+1
		net.dlossdW{m} = dXidS*net.Z{m}';
		net.dlossdb{m} = sum(dXidS,2);
		dXidS = model.weight{m}'*dXidS;
		dXidS = dXidS.*(net.Z{m} > 0);
	end
	dXidS = reshape(dXidS, model.ch_input(LC+1), []);

	for m = LC : -1 : 1
		if model.wd_subimage_pool(m) > 1
			dXidS = reshape(vTP(param, model, net, m, dXidS, 'pool_gradient'), model.ch_input(m+1), []);
		end
		net.dlossdW{m} = dXidS*net.phiZ{m}';
		net.dlossdb{m} = sum(dXidS, 2);

		if m > 1
			V = model.weight{m}' * dXidS;
			dXidS = reshape(vTP(param, model, net, m, V, 'phi_gradient'), model.ch_input(m), []);

			% vTP_pad
			a = model.ht_pad(m); b = model.wd_pad(m);
			dXidS = dXidS(:, net.idx_pad{m} + a*b*[0:num_data-1]);

			% activation function
			dXidS = dXidS.*(net.Z{m} > 0);
		end
	end
end

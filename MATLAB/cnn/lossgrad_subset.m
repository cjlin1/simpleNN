function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
net = feedforward(prob.data(:, batch_idx), model, net);
loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
	dXidS = 2*(net.Z{L+1} - Y);
	for m = L : -1 : LC+1
		net.dlossdW{m} = dXidS*net.Z{m}';
		net.dlossdb{m} = sum(dXidS, 2);
		dXidS = model.weight{m}'*dXidS;
		dXidS = dXidS.*(net.Z{m} > 0);
	end
	dXidS = reshape(dXidS, model.ch_input(LC+1), []);

	for m = LC : -1 : 1
		if model.wd_subimage_pool(m) > 1
			dXidS = vTP(model, net, m, num_data, dXidS, 'pool_gradient');
		end
		dXidS = reshape(dXidS, model.ch_input(m+1), []);
		
		if model.gpu_use
			phiZ = padding_and_phiZ(model, net, m, num_data);
			net.dlossdW{m} = dXidS*phiZ';
		else
			net.dlossdW{m} = dXidS*net.phiZ{m}';
		end
		net.dlossdb{m} = sum(dXidS, 2);

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
end

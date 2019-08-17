function [net, loss, dlossdW, dlossdb] = lossgrad_subset(prob, param, model, net, batch_idx, task)

L = param.L;
LC = param.LC;

net.num_sampled_data = length(batch_idx);
num_data = net.num_sampled_data;

Y = prob.label_mat(:, batch_idx);
dlossdW = cell(L, 1);
dlossdb = cell(L, 1);

% fun
net = feedforward(prob.data(:, batch_idx), param, model, net);
loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
	for m = 1 : L
		dlossdW{m} = zeros(size(model.weight{m}));
		dlossdb{m} = zeros(size(model.bias{m}));
	end
	ed = 0;
	inner_bsize = param.inner_bsize;
	for i = 1 : ceil(num_data/inner_bsize)
		st = ed + 1;
		ed = min(num_data, ed + inner_bsize);
		data_range = [st:ed];
		inner_num_data = ed - st + 1;
		net.num_sampled_data = inner_num_data;

		dXidS = 2*(net.Z{L+1}(:,data_range) - Y(:,data_range));
		for m = L : -1 : LC+1
			dlossdW{m} = dlossdW{m} + dXidS*net.Z{m}(:,data_range)';
			dlossdb{m} = dlossdb{m} + sum(dXidS,2);
			dXidS = model.weight{m}'*dXidS;
			dXidS = dXidS.*(net.Z{m}(:,data_range) > 0);
		end
		dXidS = reshape(dXidS, model.ch_input(LC+1), []);

		for m = LC : -1 : 1
			if model.wd_subimage_pool(m) > 1
				dXidS = vTP(param, model, net, m, dXidS, 'pool_gradient', net.idx_pool{m}(:,data_range));
			end
			dXidS = reshape(dXidS, model.ch_input(m+1), []);

			ab_data_range = to_ab_range(data_range,model.ht_input(m)*model.wd_input(m));
			phiZ = padding_and_phiZ(model, net, m, net.Z{m}(:,ab_data_range));
			dlossdW{m} = dlossdW{m} + dXidS*phiZ';
			dlossdb{m} = dlossdb{m} + sum(dXidS, 2);

			if m > 1
				V = model.weight{m}' * dXidS;
				dXidS = reshape(vTP(param, model, net, m, V, 'phi_gradient', net.idx_phiZ{m}), model.ch_input(m), []);

				% vTP_pad
				a = model.ht_pad(m); b = model.wd_pad(m);
				dXidS = dXidS(:, net.idx_pad{m} + a*b*[0:inner_num_data-1]);

				% activation function
				dXidS = dXidS.*(net.Z{m}(:,ab_data_range) > 0);
			end
		end
	end
	net.num_sampled_data = num_data;
end

function ab_data_range = to_ab_range(data_range,ab)

ab_data_range = [(data_range(1)-1)*ab+1 : data_range(end)*ab];

function net = Jacobian(param, model, net)

L = param.L;
LC = param.LC;
nL = param.nL;
num_data = net.num_sampled_data;

% Compute dzdz
net.dzdS{L} = repmat(eye(nL, nL), 1, num_data);

for m = L : -1 : LC+1
	% Compute dzdZ
	net.dzdS{m-1} = gather((model.weight{m}' * net.dzdS{m}).*reshape(repmat(net.Z{m} > 0,nL,1),[],nL*num_data));
end
% Compute dzdS or dzdZ_pool in (LC+1)th layer
net.dzdS{m-1} = reshape(net.dzdS{m-1}, [], nL, num_data);

for m = LC : -1 : 1
	if model.wd_subimage_pool(m) > 1
		net.dzdS{m} = gather(vTP(param, model, net, m, net.dzdS{m}, 'pool_Jacobian'));
	end

	net.dzdS{m} = reshape(net.dzdS{m}, model.ch_input(m+1), []);

	if m > 1
		V = gather(model.weight{m}' * net.dzdS{m});
		net.dzdS{m-1} = gather(reshape(vTP(param, model, net, m, V, 'phi_Jacobian'), model.ch_input(m), []));

		% vTP_pad
		a = model.ht_pad(m); b = model.wd_pad(m);
		net.dzdS{m-1} = net.dzdS{m-1}(:, net.idx_pad{m} + a*b*[0:nL*num_data-1]);

		net.dzdS{m-1} = reshape(net.dzdS{m-1}, [], nL, num_data) .* reshape(net.Z{m} > 0, [], 1, num_data);
	end
end

function net = Jacobian(param, model, net)

L = param.L;
LC = param.LC;
nL = param.nL;
num_data = net.num_sampled_data;
net = init_dzdS(net,model,param,num_data);

ed = 0;
inner_bsize = param.inner_bsize;
for i = 1 : ceil(num_data/inner_bsize)
    st = ed + 1;
    ed = min(num_data, ed + inner_bsize);
    data_range = [st:ed];
    inner_num_data = ed - st + 1;
    net.num_sampled_data = inner_num_data;
	nL_data_range = [(data_range(1)-1)*nL+1 : data_range(end)*nL];

	% Compute dzdz
	dzdS = repmat(eye(nL, nL), 1, inner_num_data);
	net.dzdS{L}(:,nL_data_range) = dzdS;

	for m = L : -1 : LC+1
		% Compute dzdZ
		dzdS = (model.weight{m}' * dzdS).*reshape(repmat(net.Z{m}(:,data_range) > 0,nL,1),[],nL*inner_num_data);
		if m > LC + 1
			net.dzdS{m-1}(:,nL_data_range) = dzdS;
		end
	end

	for m = LC : -1 : 1
		if model.wd_subimage_pool(m) > 1
			dzdS = vTP(param, model, net, m, dzdS, 'pool_Jacobian', net.idx_pool{m}(:,data_range));
		end

		ab_data_range = to_ab_range(data_range,nL*model.ht_input(m)*model.wd_input(m));
		dzdS = reshape(dzdS, model.ch_input(m+1), []);
		net.dzdS{m}(:,ab_data_range) = dzdS;

		if m > 1
			V = model.weight{m}' * dzdS;
			dzdS = reshape(vTP(param, model, net, m, V, 'phi_Jacobian', net.idx_phiZ{m}), model.ch_input(m), []);

			% vTP_pad
			a = model.ht_pad(m); b = model.wd_pad(m);
			dzdS = dzdS(:, net.idx_pad{m} + a*b*[0:nL*inner_num_data-1]);
			ab_data_range = to_ab_range(data_range,model.ht_input(m)*model.wd_input(m));
			dzdS = reshape(dzdS, [], nL, inner_num_data) .* reshape(net.Z{m}(:,ab_data_range) > 0, [], 1, inner_num_data);
		end
	end
end
net.num_sampled_data = num_data;

function net = init_dzdS(net,model,param,num_data)

LC = param.LC;
nL = param.nL;
for m = 1 : LC
    d = model.ch_input(m+1);
    ab = model.ht_conv(m)*model.wd_conv(m);
    net.dzdS{m} = array(zeros(d,ab*nL*num_data));
end

for m = LC+1: param.L
    num_neurons = model.full_neurons(m - LC);
    net.dzdS{m} = array(zeros(num_neurons,nL*num_data));
end

function ab_data_range = to_ab_range(data_range,ab)

ab_data_range = [(data_range(1)-1)*ab+1 : data_range(end)*ab];

function net = feedforward(data, param, model, net)

net.num_sampled_data = size(data, 2);

L = param.L;
LC = param.LC;
num_data = net.num_sampled_data;

ed = 0;
inner_bsize = param.inner_bsize;
net = initial(net, model, param, num_data);
for i = 1 : ceil(num_data/inner_bsize)
	st = ed + 1;
	ed = min(num_data, ed + inner_bsize);
	data_range = [st:ed];
	inner_num_data = ed - st + 1;
	net.num_sampled_data = inner_num_data;

	Z = array(data(:,data_range));
	Z = reshape(Z,model.ch_input(1),[]);

	ab_data_range = to_ab_range(data_range,model.ht_input(1)*model.wd_input(1));
	net.Z{1}(:,ab_data_range) = Z;
	for m = 1 : LC
		phiZ = padding_and_phiZ(model, net, m, Z);
		Z = max(model.weight{m}*phiZ + model.bias{m}, 0);
		if model.wd_subimage_pool(m) > 1
			[Z, idx_pool] = maxpooling(model, net, m, Z);
			net.idx_pool{m}(:,data_range) = idx_pool;
		end
		ab_data_range = to_ab_range(data_range,model.ht_input(m+1)*model.wd_input(m+1));
		net.Z{m+1}(:,ab_data_range) = Z;
	end

	dab = model.ch_input(LC+1) * model.wd_input(LC+1) * model.ht_input(LC+1);
	Z = reshape(Z, dab, []);

	for m = LC+1 : L-1
		Z = max(model.weight{m}*Z + model.bias{m}, 0);
		net.Z{m+1}(:,data_range) = Z;
	end

	Z = model.weight{L}*Z + model.bias{L};
	net.Z{L+1}(:,data_range) = Z;
end
net.Z{LC+1} = reshape(net.Z{LC+1},[],num_data);
net.num_sampled_data = num_data;

function net = initial(net,model,param,num_data)

LC = param.LC;
for m = 1 : LC
	d = model.ch_input(m);
	ab = model.ht_input(m)*model.wd_input(m);
	net.Z{m} = array(zeros(d,ab*num_data));
	d = model.ch_input(m+1);
	ab = model.ht_input(m+1)*model.wd_input(m+1);
	net.idx_pool{m} = array(zeros(d*ab,num_data));
end
net.Z{LC+1} = array(zeros(d,ab*num_data));

for m = LC+1: param.L
	num_neurons = model.full_neurons(m - LC);
	net.Z{m+1} = array(zeros(num_neurons,num_data));
end

function ab_data_range = to_ab_range(data_range,ab)
	
ab_data_range = [(data_range(1)-1)*ab+1 : data_range(end)*ab];


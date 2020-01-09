function vTP = vTP(model, net, m, num_data, V, op)
% output vTP: a row vector, where mat(vTP) is with dimension $d_prev a_prev b_prev \times num_v$.

nL = model.nL;

switch op
case {'pool_gradient', 'pool_Jacobian'}
	a_prev = model.ht_conv(m);
	b_prev = model.wd_conv(m);
	d_prev = model.ch_input(m+1);
	if strcmp(op, 'pool_gradient')
		num_v = num_data;
		idx = net.idx_pool{m} + [0:num_data-1]*d_prev*a_prev*b_prev;
	else
		num_v = nL*num_data;
		idx = reshape(net.idx_pool{m}, [], 1, num_data) + reshape(gpu([0:nL*num_data-1]*d_prev*a_prev*b_prev), 1, nL, num_data);
	end
case {'phi_gradient', 'phi_Jacobian'}
	a_prev = model.ht_pad(m);
	b_prev = model.wd_pad(m);
	d_prev = model.ch_input(m);

	if strcmp(op, 'phi_gradient')
		num_v = num_data;
	else
		num_v = nL*num_data;
	end

	idx = net.idx_phiZ{m}(:) + [0:num_v-1]*d_prev*a_prev*b_prev;
otherwise
	error('Unknown operation in function vTP.');
end

if model.gpu_use
	try
		vTP = accum(idx(:), V(:), [d_prev*a_prev*b_prev*num_v 1])';
	catch ME
		reset(gpuDevice(1));
		rethrow(ME);
	end
else
	vTP = accumarray(idx(:), V(:), [d_prev*a_prev*b_prev*num_v 1])';
end

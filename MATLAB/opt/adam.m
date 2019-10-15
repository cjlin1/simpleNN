function model = adam(prob, param, model, net)

beta1 = 0.9;
beta2 = 0.999;
alpha = 1e-3;
eps = 1e-8;
batch_size = param.bsize;
var_ptr = model.var_ptr;
ch_input = [model.ch_input;model.full_neurons];
num_w = var_ptr(2:end) - var_ptr(1:end-1) - ch_input(2:end);
% 1st and 2nd moment vector
M = cell(model.L,1);
V = cell(model.L,1);
for m = 1 : model.L
	M{m} = zeros(var_ptr(m+1)-var_ptr(m),1);
	V{m} = zeros(var_ptr(m+1)-var_ptr(m),1);
end

for k = 1 : param.iter_max
	for j = 1 : ceil(prob.l/batch_size) 
		batch_idx = randsample(prob.l,batch_size);
		[net, loss] = lossgrad_subset(prob, model, net, batch_idx, 'fungrad');
		for m = 1 : model.L
			gradW = [model.weight{m}(:);model.bias{m}]/param.C + [net.dlossdW{m}(:);net.dlossdb{m}]/batch_size;
			M{m} = beta1*M{m} + (1-beta1)*gradW;
			V{m} = beta2*V{m} + (1-beta2)*(gradW.*gradW);
			M_hat = M{m}/(1-beta1^k);
			V_hat = V{m}/(1-beta2^k);
			model.weight{m} = model.weight{m} - alpha*reshape(M_hat(1:num_w(m))./(sqrt(V_hat(1:num_w(m)))+eps), ch_input(m+1), []);
			model.bias{m} = model.bias{m} - alpha*(M_hat(num_w(m)+1:end)./(sqrt(V_hat(num_w(m)+1:end))+eps));
		end
	end
	fprintf('%d-epoch loss: %g\n', k, loss/batch_size);
end

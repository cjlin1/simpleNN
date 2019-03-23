function [net, f, grad] = fungrad_minibatch(prob, param, model, net, ...
                                            batch_idx, subsampled_batch, task)

if ~any(strcmp(task, {'funonly', 'fungrad'}))
	error('Unknown task.');
end

num_splits = param.num_splits;
batch_order = [1:num_splits];
% The batch for subsampled Hessian is the last to be handled
batch_order([num_splits subsampled_batch]) = batch_order([subsampled_batch num_splits]);
f = 0;
L = param.L;
LC = param.LC;
grad = [];

for batch_num = batch_order

	[net, loss] = lossgrad_subset(prob, param, model, net, batch_idx{batch_num}, task);

	f = f + loss;

	if strcmp(task, 'fungrad')
		if (batch_num == batch_order(1))
			grad.dfdW = cell(L, 1);
			grad.dfdb = cell(L, 1);
			for m = 1 : L
				grad.dfdW{m} = net.dlossdW{m};
				grad.dfdb{m} = net.dlossdb{m};
			end
		else
			for m = 1 : L
				grad.dfdW{m} = grad.dfdW{m} + net.dlossdW{m};
				grad.dfdb{m} = grad.dfdb{m} + net.dlossdb{m};
			end
		end
	end
        
end

% Obj function value and gradient norm
reg = 0.0;
for m = 1 : L
	reg = reg + norm(model.weight{m},'fro')^2 + norm(model.bias{m})^2;
	if strcmp(task, 'fungrad')
		grad.dfdW{m} = gather(model.weight{m})/param.C + grad.dfdW{m}/prob.l;
		grad.dfdb{m} = gather(model.bias{m})/param.C + grad.dfdb{m}/prob.l;
	end
end
f = (1.0/(2*param.C))*reg + f/prob.l;

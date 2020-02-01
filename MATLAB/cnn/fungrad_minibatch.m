function [net, f, grad] = fungrad_minibatch(prob, param, model, net, task)

if ~any(strcmp(task, {'funonly', 'fungrad'}))
	error('Unknown task.');
end

grad = [];
if strcmp(task, 'fungrad')
	for m = 1 : model.L
		grad.dfdW{m} = gpu(@zeros,size(model.weight{m}));
		grad.dfdb{m} = gpu(@zeros,size(model.bias{m}));
	end
end

f = 0;
bsize = param.bsize;
for i = 1 : ceil(prob.l/bsize)
	range = (i-1)*bsize + 1 : min(prob.l, i*bsize);
	[net, loss] = lossgrad_subset(prob, model, net, range, task);

	f = f + loss;

	if strcmp(task, 'fungrad')
		for m = 1 : model.L
			grad.dfdW{m} = grad.dfdW{m} + net.dlossdW{m};
			grad.dfdb{m} = grad.dfdb{m} + net.dlossdb{m};
		end
	end
end

% Obj function value and gradient vector
reg = 0.0;
for m = 1 : model.L
	reg = reg + norm(model.weight{m}, 'fro')^2 + norm(model.bias{m})^2;
	if strcmp(task, 'fungrad')
		grad.dfdW{m} = model.weight{m}/param.C + grad.dfdW{m}/prob.l;
		grad.dfdb{m} = model.bias{m}/param.C + grad.dfdb{m}/prob.l;
	end
end
f = (1.0/(2*param.C))*reg + f/prob.l;

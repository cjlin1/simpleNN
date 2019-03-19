function model = sgd(prob, param, model, net)

lr = param.lr;
batch_size = param.bsize;

for k = 1 : param.iter_max
	for j = 1 : ceil(prob.l/batch_size) 
		batch_idx = randsample(prob.l, batch_size);
		[net, loss] = lossgrad_subset(prob, param, model, net, batch_idx, 'fungrad');
		for m = 1 : param.L
			Grad = [net.dlossdW{m} net.dlossdb{m}]/batch_size;                    
			Grad = Grad + [model.weight{m} model.bias{m}]/param.C;
			model.weight{m} = model.weight{m} - lr*Grad(:,1:end-1);
			model.bias{m} = model.bias{m} - lr*Grad(:,end);
		end
	end
	fprintf('%d-epoch loss: %g\n', k, loss/batch_size);
end

model.param = param;

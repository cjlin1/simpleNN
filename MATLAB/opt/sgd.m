function best_model = sgd(prob, prob_v, param, model, net)

lr = param.lr;
decay = param.decay;

v = cell(model.L,1);
v(:) = {0};

step = 1;

best_model = model;
has_val_data = false;
if ~isempty(fieldnames(prob_v))
    has_val_data = true;
    best_val_acc = 0.0;
end

bsize = param.bsize;
for k = 1 : param.epoch_max
	for j = 1 : ceil(prob.l/bsize) 
		batch_idx = randperm(prob.l);
		[net, loss] = lossgrad_subset(prob, model, net, batch_idx(1:bsize), 'fungrad');
		for m = 1 : model.L
			Grad = [net.dlossdW{m} net.dlossdb{m}]/bsize;
			Grad = Grad + [model.weight{m} model.bias{m}]/param.C;
			v{m} = param.momentum*v{m} - lr*Grad;
			model.weight{m} = model.weight{m} + v{m}(:,1:end-1);
			model.bias{m} = model.bias{m} + v{m}(:,end);
		end
		lr = param.lr/(1 + decay*step);
		step = step + 1;
	end

	if has_val_data
        % update best_model by val_acc
        val_results = predict(prob_v, param, model, net);
        [~, val_results] = max(val_results, [], 1);
        val_acc = sum(val_results' == prob_v.y) / prob_v.l;
        if val_acc > best_val_acc
            best_model = model;
            best_val_acc = val_acc;
        end
		fprintf('%d-epoch loss/batch_size: %g val_acc: %g\n', k, loss/bsize, val_acc);
    else
        best_model = model;
		fprintf('%d-epoch loss/batch_size: %g\n', k, loss/bsize);
    end
end

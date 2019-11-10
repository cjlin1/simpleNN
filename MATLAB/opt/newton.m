function best_model = newton(prob, prob_v, param, model, net)

[net, f, grad] = fungrad_minibatch(prob, param, model, net, 'fungrad');

best_model = model;
if ~isempty(fieldnames(prob_v))
	best_val_acc = 0.0;
end

for k = 1 : param.iter_max
	if mod(k, ceil(prob.l /param.GNsize) ) == 1
		batch_idx = assign_inst_idx(param.GNsize, prob.l);
	end
	current_batch = mod(k-1, ceil(prob.l/param.GNsize)) + 1;
	[x, CGiter, gs, sGs] = CG(prob.data(:, batch_idx{current_batch}), param, model, net, grad);

	% line search
	fold = f;
	alpha = 1;
	while 1
		model = update_weights(model, alpha, x);
		prered = alpha*gs + (alpha^2)*sGs;

		[~, f, ~] = fungrad_minibatch(prob, param, model, net, 'funonly');
		actred = f - fold;
		if (actred <= param.eta*alpha*gs)
			break;
		end
		alpha = alpha * 0.5;
	end
	param = update_lambda(param, actred, prered);
	[net, f, grad] = fungrad_minibatch(prob, param, model, net, 'fungrad');

	% gradient norm
	gnorm = calc_gnorm(grad, model.L);

	if ~isempty(fieldnames(prob_v))
		% update best_model by val_acc
		val_results = predict(prob_v, param, model, net);
		[~, val_results] = max(val_results, [], 1);
		val_acc = cal_accuracy(val_results', prob_v.y);
		if val_acc > best_val_acc
			best_model = model;
			best_val_acc = val_acc;
		end
		fprintf('%d-iter f: %g |g|: %g alpha: %g ratio: %g lambda: %g #CG: %d actred: %g prered: %g val_acc: %g\n', k, f, gnorm, alpha, actred/prered, param.lambda, CGiter, actred, prered, val_acc);
	else
		best_model = model;
		fprintf('%d-iter f: %g |g|: %g alpha: %g ratio: %g lambda: %g #CG: %d actred: %g prered: %g\n', k, f, gnorm, alpha, actred/prered, param.lambda, CGiter, actred, prered);
	end
end

function param = update_lambda(param, actred, prered)

phik = actred/prered;
if (phik < 0.25)
	param.lambda = param.lambda * param.boost;
elseif (phik >= 0.75)
	param.lambda = param.lambda * param.drop;
end

function model = update_weights(model, alpha, x)

if alpha == 1
	old_alpha = 0;
else
	old_alpha = alpha / 0.5;
end

var_ptr = model.var_ptr;
channel_and_neurons = [model.ch_input; model.full_neurons];
for m = 1 : model.L
	range = var_ptr(m):var_ptr(m+1)-1;
	X = reshape(x(range), channel_and_neurons(m+1), []);   
	model.weight{m} = model.weight{m} + (alpha - old_alpha) * X(:, 1:end-1);
 	model.bias{m} = model.bias{m} + (alpha - old_alpha) * X(:, end);
end

function gnorm = calc_gnorm(grad, L)

gnorm = 0.0;

for m = 1 : L
	gnorm = gnorm + norm(grad.dfdW{m},'fro')^2 + norm(grad.dfdb{m})^2;
end
gnorm = sqrt(gnorm);

function batch_idx = assign_inst_idx(GNsize, l)

num_splits = ceil(l/GNsize);
perm_idx = randperm(l);
% ensure each subsampled Hessian has the same size
perm_idx = [perm_idx perm_idx(1:GNsize)];
batch_idx = cell(num_splits,1);

for i = 1 : num_splits
	% random selection instead of random split
	% batch_idx{i} = randperm(l, GNsize); 
	batch_idx{i} = perm_idx((i-1)*GNsize+1 : i*GNsize);
end

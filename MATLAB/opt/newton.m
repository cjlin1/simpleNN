function model = newton(prob, param, model, net)

% Assign each instance to a batch
batch_idx = assign_inst_idx(param.num_splits, prob.l);

subsampled_batch = 1;
[net, f, grad] = fungrad_minibatch(prob, param, model, net, batch_idx, ...
                                   subsampled_batch, 'fungrad');

for k = 1 : param.iter_max
	net = Jacobian(model, net);
	[x, CGiter, gs, sGs] = CG(param, model, net, grad);

	% line search
	fold = f;
	subsampled_batch = mod(k, param.num_splits)+1;
	alpha = 1;
	while 1
		model = update_weights(model, alpha, x);
		prered = alpha*gs + (alpha^2)*sGs;

		[~, f, ~] = fungrad_minibatch(prob, param, model, net, ...
                                              batch_idx, subsampled_batch, 'funonly');
		actred = f - fold;
		if (actred <= param.eta*alpha*gs)
			break;
		end
		alpha = alpha * 0.5;
	end
	param = update_lambda(param, actred, prered);
	[net, f, grad] = fungrad_minibatch(prob, param, model, net, ...
                                           batch_idx, subsampled_batch, 'fungrad');

	% gradient norm
	gnorm = calc_gnorm(grad, model.L);

	fprintf('%d-iter f: %g |g|: %g alpha: %g ratio: %g lambda: %g #CG: %d actred: %g prered: %g\n', k, f, gnorm, alpha, actred/prered, param.lambda, CGiter, actred, prered);
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


function batch_idx = assign_inst_idx(num_splits, num_data)

batch_idx = cell(num_splits, 1);
perm_idx = randi(num_splits,num_data,1);

for i = 1 : num_splits
    batch_idx{i} = find(perm_idx == i);
end

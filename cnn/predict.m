function results = predict(prob, param, model, net)

L = param.L;
l = prob.l;
l_batch = floor(param.SR * l);  % Regular batch size for each pipeline iteration
results = zeros(param.nL, l);

ed = 0;
% Though subsampled is not required in function evaluation, we use pipeline
% to save memory
for i = 1 : ceil(1/param.SR)
	% Form batch of instance sequentially
	st = ed + 1;
	ed = min(l, ed + l_batch);
	batch_idx = [st:ed];

	net = feedforward(prob.data(:, batch_idx), param, model, net);

	results(:, batch_idx) = net.Z{L+1};
end

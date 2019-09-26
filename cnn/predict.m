function results = predict(prob, SR, model, net)

L = model.L;
l = prob.l;
l_batch = floor(SR * l);  % Regular batch size for each pipeline iteration
results = zeros(model.nL, l);

ed = 0;
% Though subsampled is not required in function evaluation, we use pipeline
% to save memory
for i = 1 : ceil(1/SR)
	% Form batch of instance sequentially
	st = ed + 1;
	ed = min(l, ed + l_batch);
	batch_idx = [st:ed];

	net = feedforward(prob.data(:, batch_idx), model, net);

	results(:, batch_idx) = net.Z{L+1};
end

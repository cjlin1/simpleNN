function results = predict(prob, param, model, net)

L = model.L;
results = gpu(@zeros, [model.nL, prob.l]);

bsize = param.bsize;
for i = 1 : ceil(prob.l/bsize)
	range = (i-1)*bsize + 1 : min(prob.l, i*bsize);
	
	net = feedforward(prob.data(:, range), model, net, 'not_Jv');
	
	results(:, range) = net.Z{L+1};
end

[~, results] = max(results, [], 1);
results = model.labels(results');

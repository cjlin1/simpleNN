function [net, loss] = lossgrad_subset(prob, model, net, batch_idx, task)

L = model.L;
LC = model.LC;

num_data = length(batch_idx);

Y = gpu(@zeros, [model.nL, num_data]);
Y(prob.y_mapped(batch_idx) + model.nL*[0:num_data-1]') = 1;
%Y = prob.label_mat(:, batch_idx);

% fun
net = feedforward(prob.data(:, batch_idx), model, net, 'not_Jv');
loss = norm(net.Z{L+1} - Y, 'fro')^2;

if strcmp(task, 'fungrad')
	% grad
	v = 2*(net.Z{L+1} - Y);
	v = JTv(model, net, v);
	for m = 1 : L
    	net.dlossdW{m} = v{m}(:, 1:end-1);
    	net.dlossdb{m} = v{m}(:, end);	
	end
end

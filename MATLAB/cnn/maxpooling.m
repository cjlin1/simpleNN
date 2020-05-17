function [Zout, idx_pool, R_max_id] = maxpooling(model, net, Z, m, task)

a = model.ht_conv(m);
b = model.wd_conv(m);
d = model.ch_input(m+1);
h = model.wd_subimage_pool(m);

if exist('task', 'var') && ~any(strcmp(task, {'Jv', 'R'}))
	error('We do not support this pooling type');	
end

P = net.idx_phiZ_pool{m};
Z = reshape(Z, d*a*b, []);
Z = Z(P, :);
Z = reshape(Z, h*h, []);

if exist('task', 'var') && strcmp(task, 'R')
	Zout = Z(net.R_max_id{m});
	return;
end

[Z, max_id] = max(Z);
Zout = reshape(Z, d, []);

if exist('task', 'var') && strcmp(task, 'Jv')
	R_max_id = reshape(max_id + h*h*gpu([0:size(Z, 2)-1]), d, []);
end
outa = model.ht_input(m+1);
outb = model.wd_input(m+1);
max_id = reshape(max_id, d*outa*outb, []) + h*h*gpu([0:d*outa*outb-1])';
idx_pool = P(max_id);

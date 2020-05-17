function [Zout, idx_pool, R_Z] = maxpooling(model, net, Z, m, task)

a = model.ht_conv(m);
b = model.wd_conv(m);
d = model.ch_input(m+1);
h = model.wd_subimage_pool(m);

if exist('task', 'var') && ~any(strcmp(task, {'R'}))
	error('We do not support this pooling type');	
end

P = net.idx_phiZ_pool{m};
Z = reshape(Z, d*a*b, []);
Z = Z(P, :);
Z = reshape(Z, h*h, []);

if exist('task', 'var') && strcmp(task, 'R')
	R_Z = Z(:, end/2+1:end);
	Z = Z(:, 1:end/2);
end

[Z, max_id] = max(Z);
Zout = reshape(Z, d, []);

if exist('task', 'var') && strcmp(task, 'R')
	R_max_id = reshape(max_id + h*h*gpu([0:size(Z, 2)-1]), d, []);
	R_Z = R_Z(R_max_id);
end
outa = model.ht_input(m+1);
outb = model.wd_input(m+1);
max_id = reshape(max_id, d*outa*outb, []) + h*h*gpu([0:d*outa*outb-1])';
idx_pool = P(max_id);

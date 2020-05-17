function [Z, idx_pool, R_Z] = maxpooling(model, net, Zin, m, R_Zin)

a = model.ht_conv(m);
b = model.wd_conv(m);
d = model.ch_input(m+1);
h = model.wd_subimage_pool(m);

P = net.idx_phiZ_pool{m};
Z = reshape(Zin, d*a*b, []);
Z = Z(P, :);
Z = reshape(Z, h*h, []);

[Z, max_id] = max(Z);
Z = reshape(Z, d, []);

outa = model.ht_input(m+1);
outb = model.wd_input(m+1);
max_id = reshape(max_id, d*outa*outb, []) + h*h*gpu([0:d*outa*outb-1])';
idx_pool = P(max_id);

if exist('R_Zin', 'var')
	R_Z = reshape(R_Zin, d*a*b, []);

	idx = idx_pool + gpu([0:size(max_id, 2)-1])*d*a*b;
	R_Z = reshape(R_Z(idx), d, []);
end

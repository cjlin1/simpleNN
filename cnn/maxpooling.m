function [Zout, idx_pool] = maxpooling(model, net, m, Z)

% Z input: sigma(S_m)
a = model.ht_conv(m);
b = model.wd_conv(m);
d = model.ch_input(m+1);
h = model.wd_subimage_pool(m);

P = net.idx_phiZ_pool{m};
Z = reshape(Z, d*a*b, []);
Z = Z(P, :);
[Z, max_id] = max(reshape(Z, h*h, []));
Zout = reshape(Z, d, []);

outa = model.ht_input(m+1);
outb = model.wd_input(m+1);
max_id = reshape(max_id, d*outa*outb, []) + h*h*[0:d*outa*outb-1]';

idx_pool = array([1:d*a*b]);
idx_pool = idx_pool(P);
idx_pool = idx_pool(max_id);


function phiZ = padding_and_phiZ(model, net, Z, m, num_data)

phiZ = padding(model, Z, m, num_data);
% Calculate phiZ
phiZ = reshape(phiZ, [], num_data);
phiZ = phiZ(net.idx_phiZ{m}, :);

h = model.wd_filter(m);
d = model.ch_input(m);
phiZ = reshape(phiZ, h*h*d, []);

function output = padding(model, Z, m, num_data)

a_in = model.ht_input(m); b_in = model.wd_input(m);
a_pad = model.ht_pad(m); b_pad = model.wd_pad(m);
p = model.wd_pad_added(m);
d = model.ch_input(m);

Z = reshape(Z, d, a_in, b_in, []);
output = gpu(@zeros, [d, a_pad, b_pad, num_data]);
output(:, p+1:p+a_in, p+1:p+b_in, :) = Z;

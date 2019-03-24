function [Zout, idx_pool] = maxpooling(model, net, m)

a = model.ht_conv(m);
b = model.wd_conv(m);
d = model.ch_input(m+1);
h = model.wd_subimage_pool(m);

% Z input: sigma(S_m)
Z = net.Z{m+1};

P = net.idx_phiZ_pool{m};               %( \label{list:p-pool|P} %)
Z = reshape(Z, d*a*b, []);              %( \label{list:p-pool|gen-vec-phi-st} %)
Z = Z(P, :);                            %( \label{list:p-pool|gen-vec-phi-ed} %)
[Z, max_id] = max(reshape(Z, h*h, [])); %( \label{list:p-pool|maxpool} %)
Zout = reshape(Z, d, []);               %( \label{list:p-pool|Z-d-rows} %)

outa = model.ht_input(m+1);
outb = model.wd_input(m+1);
max_id = reshape(max_id, d*outa*outb, []) + h*h*[0:d*outa*outb-1]';  %( \label{list:p-pool|phi-pool-linearidx} %)

idx_pool = [1:d*a*b];                   %( \label{list:p-pool|Z-linear-idx} %)
idx_pool = idx_pool(P);                 %( \label{list:p-pool|phi-poolidx} %)
idx_pool = idx_pool(max_id);            %( \label{list:p-pool|poolidx} %)


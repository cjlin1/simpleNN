function phiZ = padding_and_phiZ(model, net, m)

num_data = net.num_sampled_data;
phiZ = padding(model, net, m);
% Calculate phiZ
phiZ = reshape(phiZ, [], num_data);  %( \label{list:phi|extract-phiZ-st} %)
phiZ = phiZ(net.idx_phiZ{m}, :);     %( \label{list:phi|extract-phiZ-ed} %)

h = model.wd_filter(m);
d = model.ch_input(m);
phiZ = reshape(phiZ, h*h*d, []);


function output = padding(model, net, m)

num_data = net.num_sampled_data;
a = model.ht_pad(m);
b = model.wd_pad(m);
d = model.ch_input(m);
idx = reshape(net.idx_pad{m} + [0:num_data-1]*a*b, [], 1);  %( \label{list:fun-zero-padding|offset} %)
output = zeros(d,a*b*num_data);    %( \label{list:fun-zero-padding|init-output} %)
output(:,idx) = net.Z{m};          %( \label{list:fun-zero-padding|output} %)

function [idx_pad] = find_index_padding(model, m)

a = model.ht_input(m);
b = model.wd_input(m);
p = model.wd_pad_added(m);

newa = 2*p + a;
idx_pad = reshape( gpu(p+1:p+a)' + gpu(newa*(p:p+b-1)), [], 1);

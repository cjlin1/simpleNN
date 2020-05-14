function net = init_net(model, param)

L = model.L;
LC = model.LC;
nL = model.nL;

% Initialize temporary variables for layer 1 to L+1 (used in pipeline)
net.Z = cell(L+1, 1);
net.phiZ = cell(LC, 1);
net.dlossdW = cell(L, 1);
net.dlossdb = cell(L, 1);

% P_pad, P_phi, P_pool
net.idx_phiZ = cell(LC, 1);
net.idx_phiZ_pool = cell(LC, 1);
net.idx_pool = cell(LC, 1);

bsize = param.bsize;
for m = 1 : LC
	net.idx_phiZ{m} = find_index_phiZ(model.ht_pad(m), model.wd_pad(m), model.ch_input(m), model.wd_filter(m), model.strides(m));
	net.idx_phiZ_pool{m} = find_index_phiZ(model.ht_conv(m), model.wd_conv(m), model.ch_input(m+1), model.wd_subimage_pool(m), model.wd_subimage_pool(m));

	dab = model.ch_input(m)*model.ht_pad(m)*model.wd_pad(m);
	if ~isfield(param, 'Jacobian') || ~param.Jacobian
   		net.idx_phiZ{m} = net.idx_phiZ{m}(:) + [0:bsize-1]*dab;
	else
   		net.idx_phiZ{m} = net.idx_phiZ{m}(:) + [0:nL*bsize-1]*dab;
	end
end

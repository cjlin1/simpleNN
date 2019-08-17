function u = JF_JTBJv(param, model, net, v)

L = param.L;
LC = param.LC;
nL = param.nL;
inner_bsize = param.inner_bsize;
num_data = net.num_sampled_data;
var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
u = array(zeros(n, 1));

ed = 0;
inner_bsize = param.inner_bsize;
for i = 1 : ceil(num_data/inner_bsize)
    st = ed + 1;
    ed = min(num_data, ed + inner_bsize);
    data_range = [st:ed];
    inner_num_data = ed - st + 1;
    net.num_sampled_data = inner_num_data;

    Jv = array(zeros(nL*inner_num_data, 1));
    dzdS = cell(L,1);
    dzdS{L} = repmat(eye(nL, nL), 1, inner_num_data);

    for m = L : -1 : LC+1
        var_range = var_ptr(m) : var_ptr(m+1) - 1;
        n_m = model.full_neurons(m-LC);

        p = reshape(v(var_range), n_m, []) * [net.Z{m}(:,data_range); ones(1, inner_num_data)];
        p = sum(reshape(dzdS{m}, n_m, nL, []) .* reshape(p, n_m, 1, []),1);
        Jv = Jv + p(:);

        dzdS{m-1} = (model.weight{m}' * dzdS{m}).*reshape(repmat(net.Z{m}(:,data_range) > 0,nL,1),[],nL*inner_num_data);
    end

    for m = LC : -1 : 1
        var_range = var_ptr(m) : var_ptr(m+1) - 1;
        d = model.ch_input(m+1);

        if model.wd_subimage_pool(m) > 1
            dzdS{m} = vTP(param, model, net, m, dzdS{m}, 'pool_Jacobian', net.idx_pool{m}(:,data_range));
        end
        dzdS{m} = reshape(dzdS{m}, model.ch_input(m+1), []);

        ab = model.ht_input(m)*model.wd_input(m);
        ab_data_range = [(data_range(1)-1)*ab+1 : data_range(end)*ab];
        phiZ = padding_and_phiZ(model, net, m, net.Z{m}(:,ab_data_range));

        ab = model.ht_conv(m)*model.wd_conv(m);
        p = reshape(v(var_range), d, []) * [phiZ; ones(1, ab*inner_num_data)];
        p = sum(reshape(dzdS{m}, d*ab, nL, []) .* reshape(p, d*ab, 1, []),1);
        Jv = Jv + p(:);

        if m > 1
            V = model.weight{m}' * dzdS{m};
            dzdS{m-1} = reshape(vTP(param, model, net, m, V, 'phi_Jacobian', net.idx_phiZ{m}), model.ch_input(m), []);

            a = model.ht_pad(m); b = model.wd_pad(m);
            dzdS{m-1} = dzdS{m-1}(:, net.idx_pad{m} + a*b*[0:nL*inner_num_data-1]);
            dzdS{m-1} = reshape(dzdS{m-1}, [], nL, inner_num_data) .* reshape(net.Z{m}(:,ab_data_range) > 0, [], 1, inner_num_data);
        end
    end
    % B
    Jv = 2*Jv;

    % JTv
    for m = L : -1 : LC+1
        var_range = var_ptr(m) : var_ptr(m+1) - 1;

        u_m = dzdS{m} .* Jv';
        u_m = sum(reshape(u_m, [], nL, inner_num_data), 2);
        u_m = reshape(u_m, [], inner_num_data) * [net.Z{m}(:,data_range)' ones(inner_num_data, 1)];
        u(var_range) = u(var_range) + u_m(:);
    end

    for m = LC : -1 : 1
        var_range = var_ptr(m) : var_ptr(m+1) - 1;
        d = model.ch_input(m+1);
        ab = model.ht_input(m)*model.wd_input(m);
        ab_data_range = [(data_range(1)-1)*ab+1 : data_range(end)*ab];

        phiZ = padding_and_phiZ(model, net, m, net.Z{m}(:,ab_data_range));
        u_m = reshape(dzdS{m}, [], nL*inner_num_data) .* Jv';
        u_m = sum(reshape(u_m, [], nL, inner_num_data), 2);
        u_m = reshape(u_m, d, []) * [phiZ' ones(ab*inner_num_data, 1)];
        u(var_range) = u(var_range) + u_m(:);
    end
end
u = gather(u);


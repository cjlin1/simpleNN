function [s, CGiter, gs, sGs] = CG(data, param, model, net, grad)

var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
GNsize = size(data, 2);
s = gpu(@zeros, [n, 1]);
Gv = gpu(@zeros, [n, 1]);
g = gpu(@zeros, [n, 1]);
for m = 1 : model.L
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	g(var_range) = [grad.dfdW{m}(:); grad.dfdb{m}];
end
r = -g;
v = r;

gnorm = norm(g);
rTr = gnorm^2;
cgtol = param.xi * gnorm;

if param.Jacobian
	net = Jacobian(data, param, model, net);
end

for CGiter = 1 : param.CGmax
	Gv = JTBJv(data, param, model, net, v);
	Gv = (param.lambda + 1/param.C) * v + Gv/GNsize;

	alpha = rTr / (v' * Gv);
	s = s + alpha * v;
	r = r - alpha * Gv;

	rnewTrnew = r' * r;
	% Stopping condition
	if (sqrt(rnewTrnew) <= cgtol) || (CGiter == param.CGmax)
		break
	end
	beta = rnewTrnew / rTr;
	rTr = rnewTrnew;
	v = r + beta * v;
end

% Values used for predicting function-value reduction
gs = s' * g;
sGs = s' * (-g - r - param.lambda*s);

function u = JTBJv(data, param, model, net, v)

L = model.L;
LC = model.LC;
nL = model.nL;
GNsize = size(data, 2);
var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
u = gpu(@zeros, [n, 1]);

bsize = param.bsize;
for i = 1 : ceil(GNsize/bsize)
	range = (i-1)*bsize + 1 : min(GNsize, i*bsize);
	num_data = length(range);

	% net.Z and net.phiZ not stored and must be re-calculated
	net = feedforward(data(:, range), model, net);

	% dzdS
	if param.Jacobian
		for m = 1 : L
			dzdS{m} = net.dzdS{(i-1)*L + m};
		end
	else
		dzdS = cal_dzdS(model, net, num_data);
	end

	% Jv
	p = arrayfun(@(m) Jv_one(model, net, dzdS{m}, v(var_ptr(m) : var_ptr(m+1) - 1), m, num_data), [1 : L], 'un', false);
	Jv = sum(horzcat(p{:}), 2);

	% BJv
	Jv = 2*Jv;
	
	% JTBJv
	u_m = arrayfun(@(m) JTBJv_one(model, net, dzdS{m}, Jv, m, num_data), [1 : L], 'un', false);
	u = u + vertcat(u_m{:});
end

function p = Jv_one(model, net, dzdS, v, m, num_data)

L = model.L;
LC = model.LC;
nL = model.nL;
if m >= LC + 1
	n_m = model.full_neurons(m-LC);

	p = reshape(v, n_m, []) * [net.Z{m}; ones(1, num_data)];
	p = sum(reshape(dzdS, n_m, nL, []) .* reshape(p, n_m, 1, []),1);
else
	d = model.ch_input(m+1);
	ab = model.ht_conv(m)*model.wd_conv(m);

	if model.gpu_use
		phiZ = padding_and_phiZ(model, net, m, num_data);
		p = reshape(v, d, []) * [phiZ; ones(1, ab*num_data)];
	else
		p = reshape(v, d, []) * [net.phiZ{m}; ones(1, ab*num_data)];
	end
	p = sum(reshape(dzdS, d*ab, nL, []) .* reshape(p, d*ab, 1, []),1);
end
p = p(:);

function u_m = JTBJv_one(model, net, dzdS, Jv, m, num_data)

L = model.L;
LC = model.LC;
nL = model.nL;
if m >= LC + 1
	u_m = dzdS .* Jv';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);
	u_m = reshape(u_m, [], num_data) * [net.Z{m}' ones(num_data, 1)];
else
	a = model.ht_conv(m);
	b = model.wd_conv(m);
	d = model.ch_input(m+1);

	u_m = reshape(dzdS, [], nL*num_data) .* Jv';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);

	if model.gpu_use
		phiZ = padding_and_phiZ(model, net, m, num_data);
		u_m = reshape(u_m, d, []) * [phiZ' ones(a*b*num_data, 1)];
	else
		u_m = reshape(u_m, d, []) * [net.phiZ{m}' ones(a*b*num_data, 1)];
	end
end
u_m = u_m(:);

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
	if param.Jacobian
		Gv = JTBJv(data, param, model, net, v);
	else
		Gv = R_JTBJv(data, param, model, net, v);
	end
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
num_data = size(data, 2);
var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
u = gpu(@zeros, [n, 1]);

% Jv
Jv = gpu(@zeros, [nL*num_data, 1]);
for m = L : -1 : LC+1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	n_m = model.full_neurons(m-LC);

	p = reshape(v(var_range), n_m, []);
	p = p(:, 1:end-1)*net.Z{m} + p(:, end);
	p = sum(reshape(net.dzdS{m}, n_m, nL, []) .* reshape(p, n_m, 1, []),1);
	Jv = Jv + p(:);
end

for m = LC : -1 : 1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	d = model.ch_input(m+1);
	ab = model.ht_conv(m)*model.wd_conv(m);

	p = reshape(v(var_range), d, []);
	p = p(:, 1:end-1)*net.phiZ{m} + p(:, end);
	p = sum(reshape(net.dzdS{m}, d*ab, nL, []) .* reshape(p, d*ab, 1, []),1);
	Jv = Jv + p(:);
end

% BJv
Jv = 2*Jv;

% JTBJv
for m = L : -1 : LC+1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = net.dzdS{m} .* Jv';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);

	u_m = reshape(u_m, [], num_data);
	u_m = [u_m*net.Z{m}' sum(u_m, 2)];
	u(var_range) = u(var_range) + u_m(:);
end

for m = LC : -1 : 1
	a = model.ht_conv(m);
	b = model.wd_conv(m);
	d = model.ch_input(m+1);
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = reshape(net.dzdS{m}, [], nL*num_data) .* Jv';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);

	u_m = reshape(u_m, d, []);
	u_m = [u_m*net.phiZ{m}' sum(u_m, 2)];
	u(var_range) = u(var_range) + u_m(:);
end

function u = R_JTBJv(data, param, model, net, v)

L = model.L;
LC = model.LC;
GNsize = size(data, 2);
n = model.var_ptr(end) - 1;
u = gpu(@zeros, [n, 1]);

bsize = param.bsize;
for i = 1 : ceil(GNsize/bsize)
	range = (i-1)*bsize + 1 : min(GNsize, i*bsize);

	% Jv	
	[Jv_, net] = Jv(data(:, range), model, net, v);

	% BJv
	BJv = 2*Jv_;

	% JT(BJv)
	JTv_ = JTv(model, net, BJv);
	
	JTv_ = arrayfun(@(m) JTv_{m}(:), [1 : L], 'un', 0);
	u = u + vertcat(JTv_{:});
end

function [s, CGiter, gs, sGs] = CG(param, model, net, grad)

var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
num_data = net.num_sampled_data;
s = zeros(n, 1);
Gv = zeros(n, 1);
g = zeros(n, 1);
for m = 1 : model.L
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	g(var_range) = [grad.dfdW{m}(:); grad.dfdb{m}];
end
r = -g;
v = r;

gnorm = norm(g);
rTr = gnorm^2;
cgtol = param.xi * gnorm;
for CGiter = 1 : param.CGmax
	Gv = Jv(model, net, v);
	Gv = BJv(Gv);
	Gv = JTq(model, net, Gv);
	Gv = (param.lambda + 1/param.C) * v + Gv/num_data;

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
sGs = 0.5 * s' * (-g - r - param.lambda*s);

function Jv = Jv(model, net, v)

nL = model.nL;
L = model.L;
LC = model.LC;
num_data = net.num_sampled_data;
var_ptr = model.var_ptr;
Jv = zeros(nL*num_data, 1);

for m = L : -1 : LC+1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	n_m = model.full_neurons(m-LC);

	p = reshape(v(var_range), n_m, []) * [net.Z{m}; ones(1, num_data)];
	p = sum(reshape(net.dzdS{m}, n_m, nL, []) .* reshape(p, n_m, 1, []),1);
	Jv = Jv + p(:);
end

for m = LC : -1 : 1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	ab = model.ht_conv(m)*model.wd_conv(m);
	d = model.ch_input(m+1);

	p = reshape(v(var_range), d, []) * [net.phiZ{m}; ones(1, ab*num_data)];
	p = sum(reshape(net.dzdS{m}, d*ab, nL, []) .* reshape(p, d*ab, 1, []),1);
	Jv = Jv + p(:);
end

function Jv = BJv(Jv)

Jv = 2*Jv;

function u = JTq(model, net, q)

nL = model.nL;
L = model.L;
LC = model.LC;
num_data = net.num_sampled_data;
var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
u = zeros(n, 1);

for m = L : -1 : LC+1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = net.dzdS{m} .* q';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);
	u_m = reshape(u_m, [], num_data) * [net.Z{m}' ones(num_data, 1)];
	u(var_range) = u_m(:);
end

for m = LC : -1 : 1
	a = model.ht_conv(m);
	b = model.wd_conv(m);
	d = model.ch_input(m+1);
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = reshape(net.dzdS{m}, [], nL*num_data) .* q';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);
	u_m = reshape(u_m, d, []) * [net.phiZ{m}' ones(a*b*num_data, 1)];
	u(var_range) = u_m(:);
end


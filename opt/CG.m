function [s, CGiter, gs, sGs] = CG(param, model, net, grad)

var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
num_data = net.num_sampled_data;
s = zeros(n, 1);
Gv = zeros(n, 1);
g = zeros(n, 1);
for m = 1 : param.L
	var_range = var_ptr(m) : var_ptr(m+1) - 1;
	g(var_range) = [grad.dfdW{m}(:); grad.dfdb{m}];
end
r = -g;
v = r;

gnorm = norm(g);
rTr = gnorm^2;
cgtol = param.xi * gnorm;
for CGiter = 1 : param.CGmax
	Gv = Jv(param, model, net, v);
	Gv = BJv(Gv);
	Gv = JTq(param, model, net, Gv);
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

function Jv = Jv(param, model, net, v)

nL = param.nL;
L = param.L;
LC = param.LC;
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

	p = reshape(v(var_range), d, []) * [net.phiZ{m}; ones(1, ab*num_data)];  %( \label{list:Jv|p} %)
	p = sum(reshape(net.dzdS{m}, d*ab, nL, []) .* reshape(p, d*ab, 1, []),1);  %( \label{list:Jv|Jp} %)
	Jv = Jv + p(:);  %( \label{list:Jv|sumJv} %)
end

function Jv = BJv(Jv)

Jv = 2*Jv;

function u = JTq(param, model, net, q)

nL = param.nL;
L = param.L;
LC = param.LC;
num_data = net.num_sampled_data;
var_ptr = model.var_ptr;
n = var_ptr(end) - 1;
u = zeros(n, 1);

for m = L : -1 : LC+1
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = net.dzdS{m} .* q';
	u_m = sum(reshape(u_m, [], nL, num_data), 2);
	u_m = reshape(u_m, [], num_data) * [net.Z{m}' ones(num_data, 1)];
	u(var_range) = gather(u_m(:));
end

for m = LC : -1 : 1
	a = model.ht_conv(m);
	b = model.wd_conv(m);
	d = model.ch_input(m+1);
	var_range = var_ptr(m) : var_ptr(m+1) - 1;

	u_m = reshape(net.dzdS{m}, [], nL*num_data) .* q';   %( \label{list:JTq|r-st} %)
	u_m = sum(reshape(u_m, [], nL, num_data), 2);      %( \label{list:JTq|r-ed} %)
	u_m = reshape(u_m, d, []) * [net.phiZ{m}' ones(a*b*num_data, 1)];  %( \label{list:JTq|JTq_m} %)
	u(var_range) = gather(u_m(:));
end


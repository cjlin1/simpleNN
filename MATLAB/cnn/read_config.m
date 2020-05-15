function net_config = read_config(config_file)

net_config = struct;

fid = fopen(config_file, 'r');
if fid == -1
	error('The configure file cannot be opened.');
end
while ~feof(fid)
	s = fgetl(fid);
	if ~isempty(s)
		if strcmp(s(1),'%') == 0
			eval(['net_config.' s]);
		end
	end
end
fclose(fid);

net_config.nL = net_config.full_neurons(net_config.LF);

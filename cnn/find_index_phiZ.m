function idx = find_index_phiZ(a,b,d,h,s)

first_channel_idx = ([0:h-1]*d+1)' + [0:h-1]*a*d; %( \label{list:phi|fch} %)
first_col_idx = first_channel_idx(:) + [0:d-1]; %( \label{list:phi|fcol} %)
a_out = floor((a - h)/s) + 1;  %( \label{list:phi|outa} %)
b_out = floor((b - h)/s) + 1;  %( \label{list:phi|outb} %)
column_offset = ([0:a_out-1]' + [0:b_out-1]*a)*s*d; %( \label{list:phi|co} %)
idx = column_offset(:)' + first_col_idx(:); %( \label{list:phi|idx} %)
idx = idx(:);

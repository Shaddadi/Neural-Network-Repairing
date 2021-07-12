clc
clear


all_memory_usage_hscc = [];
all_model_time_hscc = [];
% all_memory_usage = [];
% all_mode_time = [];
for n = 11:15
    filename = [num2str(n),'hscc','.mat'];
    if ~isfile(filename)
        continue;
    end
    load(filename)
    
    all_memory_usage_hscc(end+1) = single(max(all_memory_usage))/single(1024^3);
    all_model_time_hscc(end+1) = sum(all_model_time);
%     all_memory_usage(end+1) = [];
%     all_mode_time(end+1) = [];

end


VarNames = {'Tnr', 'Mnr'};
T = table(all_model_time_hscc',all_memory_usage_hscc', 'VariableNames',VarNames)

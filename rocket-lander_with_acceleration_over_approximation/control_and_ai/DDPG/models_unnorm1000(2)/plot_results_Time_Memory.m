clc
clear


all_memory_usage_hscc = [];
all_model_time_hscc = [];
all_memory_usage_new = [];
all_mode_time_new = [];
for n = 1:5
%     filename = [num2str(n),'hscc','.mat'];
%     if ~isfile(filename)
%         continue;
%     end
%     all_memory_usage_hscc(end+1) = single(max(all_memory_usage))/single(1024^3);
%     all_model_time_hscc(end+1) = sum(all_model_time);
    
    filename = [num2str(n),'hscc','.mat'];
    if ~isfile(filename)
        continue;
    end
    load(filename)
    
    all_memory_usage_new(end+1) =  single(max(all_memory_usage))/single(1024^3);
    all_mode_time_new(end+1) = sum(all_model_time);

end


VarNames = {'Tnr', 'Mnr'};
T = table(all_mode_time_new', all_memory_usage_new', 'VariableNames',VarNames)

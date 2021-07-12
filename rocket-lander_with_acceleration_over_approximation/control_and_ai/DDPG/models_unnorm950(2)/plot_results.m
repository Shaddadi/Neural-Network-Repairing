clc
clear

averaged_rewards = [];
all_times = [];
all_times_unsafety = [];
all_times_unsafety_noR = [];
all_epochs = [];
for n = 11:15
    filename = ['results',num2str(n),'.mat'];
    if ~isfile(filename)
        continue;
    end
    load(filename)
    %load([num2str(n),'.mat'])
    
    averaged_rewards(end+1) = averaged_reward;
    all_times(end+1) = all_learning_time;
    all_times_unsafety(end+1) = sum(all_computation_unsafety(:,1));
    %all_times_unsafety_noR(end+1) = sum(all_model_time);
    all_epochs(end+1) = iterations;
end

base_reward =  44.01; 
normalized_rewards = (averaged_rewards-base_reward)/base_reward;

% all_times_nr = all_times_unsafety-all_times_unsafety+all_times_unsafety_noR;
% VarNames = {'R', 'E', 'Tr', 'Tnr'};
% T = table(normalized_rewards',all_epochs',all_times',all_times_nr', 'VariableNames',VarNames)

VarNames = {'R', 'E', 'Tr'};
T = table(normalized_rewards',all_epochs',all_times', 'VariableNames',VarNames)

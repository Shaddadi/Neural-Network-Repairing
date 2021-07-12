clc
clear
epoch = 1;
load('reach_sets1.mat')
% property
all_reach_vfls = all_sets;
all_unsafe_vfls = all_unsafe_sets;



all_vfls_temp = all_reach_vfls;
unsafe_vfls_temp = all_unsafe_vfls;

% all vfls
for i = 1:size(all_vfls_temp,2)
    output_vs0 = all_vfls_temp{1,i};
    all_polys(i) = Polyhedron('V',output_vs0(:,[2,3]));
end

% unsafe vfls
if ~isempty(unsafe_vfls_temp)
    for i = 1:size(unsafe_vfls_temp,2)
        output_vs1 = unsafe_vfls_temp{1,i};
        unsafe_polys(i) = Polyhedron('V',output_vs1(:,[2,3]));
    end
else
    unsafe_polys = {};
end

% plot

fig = figure();
xlim([-45, 35])
ylim([-25, 70])
ax = gca;
ax.FontSize = 8; 
ax2 = gcf;
ax2.Position = [500, 400, 300, 250];

all_polys.plot('edgealpha',0.0,'color','b','alpha',1.0)
hold on 
if ~isempty(unsafe_polys)
    unsafe_polys.plot('edgealpha',0.0,'color','r','alpha',1.0)
end
hold off

xlim([-45, 35])
ylim([-25, 70])

savefig(['Output Reachable Domain on y_2 and y_3 : Epoch ',num2str(epoch),'.fig'])
saveas(fig, ['Output Reachable Domain on y_2 and y_3 : Epoch ',num2str(epoch),'.png'])
close all
close all hidden




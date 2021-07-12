clc
clear
load('reach_sets.mat')
boundary = 1;
% property
for p = 1:2
    for epoch = 1

        all_vfls_temp = all_reach_vfls{epoch, p};
        unsafe_vfls_temp = all_unsafe_vfls{epoch, p};

        % all vfls
        for i = 1:size(all_vfls_temp,1)
            vs = all_vfls_temp{i,1};
            M = all_vfls_temp{i,2};
            b = all_vfls_temp{i,3};
            output_vs0 = vs*M'+b';
            all_polys(i) = Polyhedron('V',output_vs0);
        end

        % unsafe vfls
        if ~isempty(unsafe_vfls_temp)
            for i = 1:size(unsafe_vfls_temp,1)
                vs = unsafe_vfls_temp{i,1};xlim([0 5])
                M = unsafe_vfls_temp{i,2};
                b = unsafe_vfls_temp{i,3};
                output_vs1 = vs*M'+b';
                unsafe_polys(i) = Polyhedron('V',output_vs1);
            end
        else
            unsafe_polys = {};
        end

        % plot
        fig = figure();
        all_polys.plot('edgealpha',0.0,'color','b','alpha',1.0)
        hold on 
        if ~isempty(unsafe_polys)
            unsafe_polys.plot('edgealpha',0.0,'color','r','alpha',1.0)
        end
%         plot(outputs_p2(:,1), outputs_p2(:,2),'.g')
        hold off
        xlabel('y_0')
        ylabel(['y_',num2str(boundary)])
%             xlim([-0.01 0.05])
%             ylim([-0.03, 0.02])
        title(['Output Reachable Domain on Property',num2str(p),': Epoch ',num2str(epoch)])
        savefig(['Output Reachable Domain on Property',num2str(p),': Epoch ',num2str(epoch),'.fig'])
        saveas(fig, ['Output Reachable Domain on Property',num2str(p),': Epoch ',num2str(epoch),'.png'])
        close all

     end
    
end

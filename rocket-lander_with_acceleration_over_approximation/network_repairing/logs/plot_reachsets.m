clc
clear
load('reach_sets.mat')
% property
for p = 1
    for epoch = 1:36 %1:length(all_reach_vfls)
        for boundary = 1
            all_vfls_temp = all_reach_vfls{epoch, p};
            unsafe_vfls_temp = all_unsafe_vfls{epoch, p};

            % all vfls
            for i = 1:size(all_vfls_temp,1)
                vs = all_vfls_temp{i,1};
                M = all_vfls_temp{i,2};
                b = all_vfls_temp{i,3};
                output_vs0 = vs*M'+b';
                all_polys(i) = Polyhedron('V',output_vs0(:,[1,boundary+1]));
            end

            % unsafe vfls
            if ~isempty(unsafe_vfls_temp)
                for i = 1:size(unsafe_vfls_temp,1)
                    vs = unsafe_vfls_temp{i,1};xlim([0 5])
                    M = unsafe_vfls_temp{i,2};
                    b = unsafe_vfls_temp{i,3};
                    output_vs1 = vs*M'+b';
                    unsafe_polys(i) = Polyhedron('V',output_vs1(:,[1,boundary+1]));
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
            hold off
            xlabel('y_0')
            ylabel(['y_',num2str(boundary)])
            xlim([-200, 150])
            ylim([-200, 150])
            title(['Output Reachable Domain on y_0 and y_',num2str(boundary),': Epoch ',num2str(epoch)])
            savefig(['Output Reachable Domain on y_0 and y_',num2str(boundary),': Epoch ',num2str(epoch),'.fig'])
            if epoch<10
                saveas(fig, ['Output Reachable Domain on y_0 and y_',num2str(boundary),': Epoch 00',num2str(epoch),'.png'])
            elseif epoch<100
                saveas(fig, ['Output Reachable Domain on y_0 and y_',num2str(boundary),': Epoch 0',num2str(epoch),'.png'])
            else
                saveas(fig, ['Output Reachable Domain on y_0 and y_',num2str(boundary),': Epoch ',num2str(epoch),'.png'])
            end
            
            clear all_polys
            clear unsafe_polys
            close all
        end
     end
    
end

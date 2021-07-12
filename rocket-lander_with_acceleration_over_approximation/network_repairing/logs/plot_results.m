clc
clear
load('all_property_result.mat')
load('all_test_accuracy.mat')

x = 1:length(all_test_accuracy);
fig = figure;

subplot(3,1,1)
plot(x, all_test_accuracy,'-')
grid on
ylabel('Accuracy')
ylim([0.98 1.0])

subplot(3,1,2)
hold on
p = all_property_result(:,1);
plot(x, p,'-.')
ylabel({'P1 Unsafe Volume'})
indx0 = find(p==-1.0e-4);
indx1 = find(p~=-1.0e-4);
plot(x, p,'-')
plot(x(indx0), p(indx0),'.b')
plot(x(indx1), p(indx1),'.r')
grid on
hold off

subplot(3,1,3)
hold on
p = all_property_result(:,2);
plot(x, p,'-.')
ylabel({'P2 Unsafe Volume'})
indx0 = find(p==-1.0e-4);
indx1 = find(p~=-1.0e-4);
plot(x, p,'-')
plot(x(indx0), p(indx0),'.b')
plot(x(indx1), p(indx1),'.r')
grid on
hold off


xlabel('Epoch')

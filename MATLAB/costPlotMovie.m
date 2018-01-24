function costPlotMovie(state_gp,input_gp,state_gs,input_gs,state_idd,input_idd,s,c)
% prepare the visualization window and graphics callback

% grid on
% box on
% hold on
fig = figure(1);
handles = []; handles2 = []; handles3 = [];
writeObj = VideoWriter(c);
writeObj.FrameRate = 10;
open(writeObj);
lW = 1.2;
% set(0,'currentfigure',10);
cMap = colormap('lines');
set(gca,'fontsize',15)
% set(gcf,'Position',[440 378 550 220])
plot(state_gp(1,end),state_gp(2,end),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(state_gs(1,end),state_gs(2,end),'-.','color',cMap(2,:),'linewidth',lW)
plot(state_idd(1,end),state_idd(2,end),':','color','k','linewidth',2)
leg = legend('Sparse GP','GKS','IDD','Location','southeast');
legend boxoff
N = size(state_gp,2);
c_gp = zeros(1,N);
c_gs = zeros(1,N);
c_idd = zeros(1,N);
t = 0:0.1:60;
for i=2:1:N
    %
    if s == 'c'
    [cost_gp,~,~] = costC(state_gp(:,1:i),input_gp(:,1:i-1),1.4,0.7,1.2);
    [cost_gs,~,~] = costC(state_gs(:,1:i),input_gs(:,1:i-1),1.4,0.7,1.2);
    [cost_idd,~,~] = costC(state_idd(:,1:i),input_idd(:,1:i-1),1.4,0.7,1.2);
    c_gp(1,i-1) = sum(cost_gp);
    c_gs(1,i-1) = sum(cost_gs);
    c_idd(1,i-1) = sum(cost_idd);
    axis([0 60 0 1000])
    elseif s == 'tr'
    [cost_gp,~,~] = costTr(state_gp(:,1:i),input_gp(:,1:i-1),1.5,1,2.4);
    [cost_gs,~,~] = costTr(state_gs(:,1:i),input_gs(:,1:i-1),1.5,1,2.4);
    [cost_idd,~,~] = costTr(state_idd(:,1:i),input_idd(:,1:i-1),1.5,1,2.4);
    c_gp(1,i-1) = sum(cost_gp);
    c_gs(1,i-1) = sum(cost_gs);
    c_idd(1,i-1) = sum(cost_idd);
    axis([0 60 0 1600])
    elseif s == 'sq'
    [cost_gp,~,~] = costSq(state_gp(:,1:i),input_gp(:,1:i-1),1.5,0.6,2.4);
    [cost_gs,~,~] = costSq(state_gs(:,1:i),input_gs(:,1:i-1),1.5,0.6,2.4);
    [cost_idd,~,~] = costSq(state_idd(:,1:i),input_idd(:,1:i-1),1.5,0.6,2.4);
    c_gp(1,i-1) = sum(cost_gp);
    c_gs(1,i-1) = sum(cost_gs);
    c_idd(1,i-1) = sum(cost_idd);
    axis([0 60 0 1600])
    end
    %
    delete(handles)
    delete(handles2) 
    delete(handles3)
    %
    hold on
    handles3 = plot(t(2:i),c_idd(1:i-1),':','linewidth',1.8,'color','k' );
    handles2 = plot(t(2:i),c_gs(1:i-1),'-.','linewidth',1.2,'color',cMap(2,:) );
    handles = plot(t(2:i),c_gp(1:i-1),'-','linewidth',1.2,'color',cMap(1,:) );
    xlabel('time (s)')
    title('cost')
    grid on
    %
    set(gca,'fontsize',15)
    drawnow
    writeVideo(writeObj,getframe(fig));
end

close(writeObj);
end


function carMovie(x1,simTime,delT,c)
% prepare the visualization window and graphics callback
% set(gcf,'name','car parking','Menu','none','NumberT','off')
% set(gca,'xlim',[-30 30],'ylim',[-5 40],'DataAspectRatio',[1 1 1])


figure()
grid on
axis([-1 1 -1 1]*6), hold on
axis square
% plot target configuration with light colors
% handles = carPlot(x1(:,1), 3);
% fcolor  = get(handles,'facecolor');
% ecolor  = get(handles,'edgecolor');
% fcolor  = cellfun(@(x) (x+3)/4,fcolor,'UniformOutput',false);
% ecolor  = cellfun(@(x) (x+3)/4,ecolor,'UniformOutput',false);
% set(handles, {'facecolor','edgecolor'}, [fcolor ecolor])

% prepare and install trajectory visualization callback
line_handle = line([0 0],[0 0],'color','b','linewidth',2);
plotFn = @(x) set(line_handle,'Xdata',x(1,:),'Ydata',x(2,:));
Op.plotFn = plotFn;

handles = []; target = [];
% fig = figure(1);
% writeObj = VideoWriter(c);
% writeObj.FrameRate = 10;
% open(writeObj);
for i=1:1:simTime/delT

    delete(handles)
%     delete(target)
    handles = carPlot(x1(:,i),3);
%     target = plot(0,3,'o','color','r','markerSize',15);
    drawnow
%     writeVideo(writeObj,getframe(fig));
end

% close(writeObj);
fprintf('finish\n')
end


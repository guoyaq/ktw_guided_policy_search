clear all; close all; clc;
delT = 0.1;
horizon = 150;
simTime = delT * horizon;
x0 = [2;-2;pi/2];
u0 = zeros(2,horizon);
t = 0:delT:horizon*delT;
% DDP starts
[xDDP,uDDP,cost,trace,Quu,QuuF] = ptDDPverOne(x0,u0,delT);
%% MPC starts
horizon = 30;
u0_MPC = ones(2,horizon);
xMPC = []; uMPC = []; ctMPC = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    tic
    if i == 1
        xCu = x0;
        [~,uR] = ptDDPverOne(xCu,u0_MPC,delT);
    else
        xCu = xTemp;
        [~,uR] = ptDDPverOne(xCu,uR,delT);
    end
    ctMPC(i) = toc;
    uTemp = uR(:,1);
    uMPC(:,i) = uTemp;
    % Dynamics
    xTemp = ptDyn(xCu,uMPC(:,i),delT);
    xMPC(:,i) = xTemp;
end
xMPC = [x0,xMPC];
%% NN starts
k = 9;
% w1 = load(['w1_',num2str(k),'.txt']);
% w2 = load(['w2_',num2str(k),'.txt']);
% b1 = load(['b1_',num2str(k),'.txt']);
% b2 = load(['b2_',num2str(k),'.txt']);
% var = load(['var_',num2str(k),'.txt']);
x0 = [1;-2;pi/3];
w1 = load('w1.txt');
b1 = load('b1.txt');
w2 = load('w2.txt');
b2 = load('b2.txt');
w3 = load('w3.txt');
b3 = load('b3.txt');
% var = load('var.txt');
xNN = []; uNN = []; ctNN = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    tic
    if i == 1
        xCu = x0;
        %         uR = singleNN(xCu,w_input_hidden,w_hidden_output);
        uR = doubleNN_tf(xCu,w1,w2,w3,b1,b2,b3);
    else
        xCu = xTemp;
        %         uR = singleNN(xCu,w_input_hidden,w_hidden_output);
        uR = doubleNN_tf(xCu,w1,w2,w3,b1,b2,b3);
    end
    ctNN(i) = toc;
    uTemp = uR;
    %     uTemp = mvnrnd(uR,var);
    uNN(:,i) = uTemp;
    % Dynamics
    xTemp = ptDyn(xCu,uNN(:,i),delT);
    xNN(:,i) = xTemp;
end
xNN = [x0,xNN];
carMovie(xNN,simTime,delT,'final')
%% cost
cDDP = ptCost(xDDP,[uDDP,zeros(2,1)]);
cMPC = ptCost(xMPC,[uMPC(1:2,:),zeros(2,1)]);
cNN = ptCost(xNN,[uNN(1:2,:),zeros(2,1)]);
fprintf('Cost for DDP = %d\n',sum(cDDP))
fprintf('Cost for MPC = %d\n',sum(cMPC))
fprintf('Cost for NN = %d\n',sum(cNN))
%%
carMovie(xDDP,simTime,delT,'DDP')
% carMovie(xNN,simTime,delT,'final')
%%
figure(1)
cMap = colormap('lines');
lW = 1.1;
fS = 15;
plot(xDDP(1,:),xDDP(2,:),'color',cMap(1,:),'linewidth',lW)
axis([-1 1 -1 1]*6)
xlabel('X[m]')
ylabel('Y[m]')
grid on
set(gca,'fontsize',fS)
%%
figure(2)
cMap = colormap('lines');
lW = 1.1;
fS = 15;
subplot(1,2,1)
plot(t(1:end-1),uNN(1,:),'color',cMap(1,:),'linewidth',lW), hold on
% plot(t(1:end-1),u_result(1,:),'--','color',cMap(2,:),'linewidth',lW)
title('v')
grid on
subplot(1,2,2)
plot(t(1:end-1),uNN(2,:),'color',cMap(1,:),'linewidth',lW), hold on
% plot(t(1:end-1),u_result(2,:),'--','color',cMap(2,:),'linewidth',lW)
title('w')
grid on
%%
figure(3)
subplot(1,3,1)
plot(t,x(1,:),'color',cMap(1,:),'linewidth',lW), hold on
% plot(t,x_result(1,:),'--','color',cMap(2,:),'linewidth',lW)
title('x')
grid on
subplot(1,3,2)
plot(t,x(2,:),'color',cMap(1,:),'linewidth',lW), hold on
% plot(t,x_result(2,:),'--','color',cMap(2,:),'linewidth',lW)
title('y')
grid on
subplot(1,3,3)
plot(t,x(3,:),'color',cMap(1,:),'linewidth',lW), hold on
% plot(t,x_result(3,:),'--','color',cMap(2,:),'linewidth',lW)
title('yaw')
grid on
%% 
figure(4)
plot(1:10,cNN_T,'color',cMap(2,:),'linewidth',lW), hold on
plot(0:10,[0:10] * 0 +sum(cDDP),'--','color',cMap(1,:),'linewidth',lW)
legend('GPS results','iLQR baseline'), legend boxoff
plot(1:10,cNN_T,'o','color',cMap(2,:),'linewidth',lW,'markerSize',10)
title('cost per iteration')
xlabel('iteration of GPS')
ylabel('cost')
set(gca,'fontsize',fS)
grid on



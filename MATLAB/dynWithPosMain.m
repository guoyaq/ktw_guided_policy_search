clear all; close all; clc;
delT = 0.01;
simTime = 2;
t = 0:delT:simTime;
x0 = [-pi/2;0;0;0];
u0_DDP = ones(2,simTime / delT);
xG = [0.5;1;0;0];
model_e = 1;
%% DDP start
fprintf('DDP starts\n')
[~,uDDP] = dynWithDDP(x0,u0_DDP,delT,xG);
xDDP = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    if i == 1
        xCu = x0;
    else
        xCu = xTemp;
    end
    % Dynamics
    xTemp = dynWithPosError(xCu,uDDP(:,i),delT,model_e,model_e);
    xDDP(:,i) = xTemp;
end
xDDP = [x0,xDDP];
fprintf('DDP is finished\n')
%% MPC start
horizon = 20;
u0_MPC = ones(2,horizon);
xMPC = []; uMPC = []; ctMPC = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    tic
    if i == 1
        xCu = x0;
        [~,uR] = dynWithDDP(xCu,u0_MPC,delT,xG);
    else
        xCu = xTemp;
        [~,uR] = dynWithDDP(xCu,uR,delT,xG);
    end
    ctMPC(i) = toc;
    uTemp = uR(:,1);
    uMPC(:,i) = uTemp;
    % Dynamics
    xTemp = dynWithPosError(xCu,uMPC(:,i),delT,model_e,model_e);
    xMPC(:,i) = xTemp;
end
xMPC = [x0,xMPC];
%% NN start
w1 = load('w1.txt');
w2 = load('w2.txt');
b1 = load('b1.txt');
b2 = load('b2.txt');
xNN = []; uNN = []; ctNN = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    tic
    if i == 1
        xCu = x0;
%         uR = singleNN(xCu,w_input_hidden,w_hidden_output);
        uR = singleNN_tf(xCu,w1,w2,b1,b2);
    else
        xCu = xTemp;
%         uR = singleNN(xCu,w_input_hidden,w_hidden_output);
        uR = singleNN_tf(xCu,w1,w2,b1,b2);
    end
    ctNN(i) = toc;
    uTemp = uR;
%     uTemp = mvnrnd(uR,[110,26;26,10]);
    uNN(:,i) = uTemp;
    % Dynamics
    xTemp = dynWithPosError(xCu,uNN(:,i),delT,model_e,model_e);
    xNN(:,i) = xTemp;
end
xNN = [x0,xNN];
%% cost
cDDP = dynWithCost(xDDP,[uDDP,zeros(2,1)],xG);
    cMPC = dynWithCost(xMPC,[uMPC,zeros(2,1)],xG);
cNN = dynWithCost(xNN,[uNN(1:2,:),zeros(2,1)],xG);
fprintf('Cost for DDP = %d\n',sum(cDDP))
fprintf('Cost for MPC = %d\n',sum(cMPC))
fprintf('Cost for NN = %d\n',sum(cNN))
%%
lW = 2;
fS = 18;
figure()
cMap = colormap('lines');
subplot(121)
plot(t,xDDP(1,:),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t,xMPC(1,:),'-','color',cMap(2,:),'linewidth',lW)
plot(t,xNN(1,:),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('theta1')
xlabel('time (s)')
ylabel('rad')
grid on
set(gca,'fontsize',fS)
subplot(122)
plot(t,xDDP(2,:),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t,xMPC(2,:),'-','color',cMap(2,:),'linewidth',lW)
plot(t,xNN(2,:),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('theta2')
xlabel('time (s)')
ylabel('rad')
grid on
set(gca,'fontsize',fS)
figure()
subplot(121)
plot(t(1:end-1),uDDP(1,:),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t(1:end-1),uMPC(1,:),'-','color',cMap(2,:),'linewidth',lW)
plot(t(1:end-1),uNN(1,:),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('input1')
xlabel('time (s)')
ylabel('Nm')
grid on
set(gca,'fontsize',fS)
subplot(122)
plot(t(1:end-1),uDDP(2,:),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t(1:end-1),uMPC(2,:),'-','color',cMap(2,:),'linewidth',lW)
plot(t(1:end-1),uNN(2,:),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('input2')
xlabel('time (s)')
ylabel('Nm')
set(gca,'fontsize',fS)
grid on
%%
figure()
plot(t,cDDP(1,:),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t,cMPC(1,:),'-','color',cMap(2,:),'linewidth',lW)
plot(t,cNN(1,:),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('Cost')
xlabel('time (s)')
ylabel('Cost')
set(gca,'fontsize',fS)
grid on
%%
beta = 100;
figure()
plot(t(1,end-beta:end),cDDP(1,end-beta:end),'-','color',cMap(1,:),'linewidth',lW), hold on
plot(t(1,end-beta:end),cMPC(1,end-beta:end),'-','color',cMap(2,:),'linewidth',lW)
plot(t(1,end-beta:end),cNN(1,end-beta:end),'-','color',cMap(3,:),'linewidth',lW)
legend('iLQR','MPC','NN')
legend boxoff
title('Cost')
xlabel('time (s)')
ylabel('Cost')
set(gca,'fontsize',fS)
grid on
%%
doublePendulumMovie(xDDP,simTime/delT,[1;1],xG,'Temp')
%%
doublePendulumMovie(xMPC,simTime/delT,[1;1],xG,'Temp')
%%
doublePendulumMovie(xNN,simTime/delT,[1;1],xG,'Temp')



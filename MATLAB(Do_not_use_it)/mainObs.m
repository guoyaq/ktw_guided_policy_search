clear all; close all; clc;
delT = 0.1;
horizon = 150;
simTime = delT * horizon;
x_t = [0;3];
u0 = zeros(2,horizon);
t = 0:delT:horizon*delT;
%% NN starts
k = 9;
% w1 = load(['w1_',num2str(k),'.txt']);
% w2 = load(['w2_',num2str(k),'.txt']);
% b1 = load(['b1_',num2str(k),'.txt']);
% b2 = load(['b2_',num2str(k),'.txt']);
% var = load(['var_',num2str(k),'.txt']);
x0 = [4;-2;pi/2];
w1 = load('../w1.txt');
b1 = load('../b1.txt');
w2 = load('../w2.txt');
b2 = load('../b2.txt');
w3 = load('../w3.txt');
b3 = load('../b3.txt');
% var = load('var.txt');
xNN = []; uNN = []; ctNN = [];
for i = 1 : simTime/delT
    fprintf('Current time = %d\n',i*delT)
    tic
    if i == 1
        xCu = x0;
        o = getObs(x_t,xCu);
        uR = doubleNN_tf(o,w1,w2,w3,b1,b2,b3);
    else
        xCu = xTemp;
        o = getObs(x_t,xCu);
        uR = doubleNN_tf(o,w1,w2,w3,b1,b2,b3);
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
carMovie(xNN,simTime,delT,'right')
%%
cNN = ptCost(xNN,[uNN(1:2,:),zeros(2,1)]);
fprintf('Cost for NN = %d\n',sum(cNN))
%%
% carMovie(xNN,simTime,delT,'final')
%% action
kb = HebiKeyboard();
x0 = [4;-2;pi/3];
x_t = [0;3];
dx = 0.2;
%
figure()
grid on
axis([-1 1 -1 1]*6), hold on
axis square
%
car = [];
target = [];
xCu = x0;
flag = true;

fig = figure(1);
% writeObj = VideoWriter('moving');
% writeObj.FrameRate = 10;
% open(writeObj);

while true
    
    state = read(kb);
    
    if all(state.keys('w'))
        disp('w is both pressed!')
        x_t(2) = x_t(2) + dx;
    end
    if all(state.keys('a'))
        disp('a is both pressed!')
        x_t(1) = x_t(1) - dx;
    end
    if all(state.keys('s'))
        disp('s is both pressed!')
        x_t(2) = x_t(2) - dx;
    end
    if all(state.keys('d'))
        disp('d is both pressed!')
        x_t(1) = x_t(1) + dx;
    end
    
    

    o = getObs(x_t,xCu);
    uR = doubleNN_tf(o,w1,w2,w3,b1,b2,b3);
    xCu = ptDyn(xCu,uR,delT);
    
    delete(car)
    delete(target)
    car = carPlot(xCu,3);
    target = plot(x_t(1),x_t(2),'o','color','r','markerSize',15);
    drawnow
%     writeVideo(writeObj,getframe(fig));
    while flag
        state2 = read(kb);
        if all(state2.keys('h'))
            disp('h is both pressed!')
            flag = false;
            break
        end
        pause(0.1);
    end
    
       
    
%     pause(0.1);
end
% close(writeObj);
fprintf('finish\n')









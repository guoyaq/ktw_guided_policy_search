function [x,u,cost,trace,Quu,QuuF] = ptDDPverOne(initialState,initialInput,delT)

full_DDP = false;

% the optimization path
DYNCST = @(state,input) ptCst(state,input,full_DDP,delT);

Op.lims = [];

[x,u,~,~,~,~,cost,trace,~,Quu,QuuF]= myiLQGverOne(DYNCST, initialState, initialInput, Op);



end


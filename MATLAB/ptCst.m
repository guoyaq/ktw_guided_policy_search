function [f,c,fx,fu,fxx,fxu,fuu,cx,cu,cxx,cxu,cuu] = ptCst(state,input,full_DDP,delT)
% combine car dynamics and cost
% use helper function finite_difference() to compute derivatives
if nargout == 2
   f = ptDyn(state,input,delT);
   c = ptCost(state,input);
else
   % state and control indices
   ix = 1:3;
   iu = 4:5;

   % dynamic first derivatives
   xu_dyn = @(xu) ptDyn(xu(ix,:), xu(iu,:),delT);
   J = finite_difference(xu_dyn,[state;input]);
%    J = ptFirstDeriv([state;input],delT);
   fx = J(:,ix,:);
   fu = J(:,iu,:);
   
   % dynamics second derivatives
   % none
   [fxx,fxu,fuu] = deal([]);
   
   % cost first derivatives
%    I = repmat(i,1,n+1);
   xu_cost = @(xu) ptCost(xu(ix,:),xu(iu,:));
   J = squeeze(finite_difference(xu_cost,[state;input]));
   cx = J(ix,:);
   cu = J(iu,:);
   
   % cost second derivatives
%    I2 = repmat(I,1,n+1);
   xu_cost2 = @(xu) ptCost(xu(ix,:),xu(iu,:));
   xu_Jcst = @(xu) squeeze(finite_difference(xu_cost2,xu));
   JJ = finite_difference(xu_Jcst,[state;input]);
   JJ = 0.5*(JJ + permute(JJ,[2 1 3]));
   cxx = JJ(ix,ix,:);
   cxu = JJ(ix,iu,:);
   cuu = JJ(iu,iu,:);
   
   [f,c] = deal([]);
    
    
end

end
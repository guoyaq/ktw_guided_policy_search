function o = getObs(P_t,P_c)

% x_d = P_t(1);
% y_d = P_t(2);

x = P_c(1);
y = P_c(2);
yaw = P_c(3);

R = [cos(yaw),-sin(yaw); ...
     sin(yaw),cos(yaw)];
 
trans = [x;y];

H = [R',-R'*trans;0,0,1];

P_T = P_t;
P_T(3) = 1;

P_t_local = H * P_T;
yaw_diff = atan2(P_t_local(2),P_t_local(1));

o = [P_t_local(1:2);yaw_diff];


end


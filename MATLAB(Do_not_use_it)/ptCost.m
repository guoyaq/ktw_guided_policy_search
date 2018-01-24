function c = ptCost(state,input)
% lu: quadratic cost on controls
% lf: final cost on distance from target parking configuration
% lx: running cost on distance from origin to encourage tight turns

final = isnan(input(1,:));
input(:,final) = 0;

% control cost
lc = 1/2 * 1 * (  1*input(1,:).^2 + 1*input(2,:).^2  ) ;

% state cost
lx = 1/2 * 1e-1 * (  (state(1,:) - 0).^2 + ...
                     (state(2,:) - 2).^2 + ...
                     (state(3,:) - pi / 2).^2 );

% total cost

c =  lx + lc;

end
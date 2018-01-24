function  output = singleNN_tf( input,w1,w2,b1,b2 )

% x = [1;input];
s_j = w1' * input + b1;
z_j = softPlus(s_j);
z = z_j;
s_k = w2' * z + b2;
output = linearFun(s_k);

end

function y = softPlus(x)

y = log(1 + exp(x));

end

function y = linearFun(x)

y = x;

end



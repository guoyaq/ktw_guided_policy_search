function  output = doubleNN_tf( input,w1,w2,w3,b1,b2,b3)

% x = [1;input];
s_1 = w1' * input + b1;
z_1 = relu(s_1);
s_2 = w2' * z_1 + b2;
z_2 = relu(s_2);
s_3 = w3' * z_2 + b3;
output = linearFun(s_3);

end

function y = softPlus(x)

y = log(1 + exp(x));

end

function y = linearFun(x)

y = x;

end

function y = relu(x)

x_test = x > 0;

y = x_test .* x;

end


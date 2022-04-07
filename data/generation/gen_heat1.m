N = 120;
s = 1024;
dt = 0.05;
T = 1;
c = 0.05;
tspan = [0:dt:T];
% parameters for the Gaussian random field
gamma = 2.5;
tau = 7;
sigma = 7^(2);

x = linspace(0,1,s+1);

input = zeros(N, s);
output = zeros(N, length(tspan), s);

h = waitbar(0,'Please wait...');
for j=1:N
    constant = normrnd(0, 0.01);
    u0 = GRF(s/2, 0, gamma, tau, sigma)+constant;
    u = heat1(u0, tspan, c);
    
    u0eval = u0(x);
    input(j,:) = u0eval(1:end-1);
    for k=1:(length(tspan))
        ueval=u{k}(x);
        output(j,k,:) = ueval(1:end-1);
    end
    %waitbar(j/N);
    waitbar(j/N, h, j);
end
close(h); clear h;
save('heat_data_1024.mat','input','output');
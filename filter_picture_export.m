clear;
close all;
tic;
dilation_rate = 0;
config = "proj6-filters";
mkdir(strcat("filter_pictures/",config));
mkdir(strcat("filter_pictures/",config,"/ex"));
mkdir(strcat("filter_pictures/",config,"/in"));
A = imread("Project/Lenna128.png");
% A = imread("Project/lenna64.png");
% A = imread("Project/cityscapes-smol.jpeg");
% grayscales
% B = (0.3 * A(:,:,1) + 0.4 * A(:,:,2) + 0.1 * A(:,:,3));
B = (0.2126 * A(:,:,1) + 0.7152 * A(:,:,2) + 0.0722 * A(:,:,3));

D = zeros(int16(size(B,1)),int16(size(B,2)));

% Initial Values / Constants
C = 1;
E_L = 10.613;
E_K = -12;
E_Na = 115;
G_L = 0.3;
G_K = 36;
G_Na = 120;

V_th = -65;
V_reset = -70;
E_ex = 0;
E_in = -75;
tau_ex = 3;
tau_in = 10;
A_ex = 0.028953;
A_in = 0.014103;
tau_ref = 10;

% I = @(x,y) B(x + int16(size(B,2)/2) + 1,int16(size(B,1)) - (y + int16(size(B,1)/2) - 1));
I = @(x,y) B(x,y);
d2r = @(d) d * (pi / 180);

% before
% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+62)/10));
% n_inf=@(v) 1./(1+exp(-(v+53)/16));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+62)/10));
% tau_n=@(v) 1+6./(1+exp((v+53)/16));

m_inf=@(v) 1./(1+exp(-(v+40)/9));
h_inf=@(v) 1./(1+exp((v+62)/15));
n_inf=@(v) 1./(1+exp(-(v+20)/30));
tau_m=@(v) 0.3 + 0.0*v;
tau_h=@(v) 1+11./(1+exp((v+62)/10));
tau_n=@(v) 1+6./(1+exp((v+53)/16));

theta_i = 0;
normal = [-double(cos(d2r(theta_i))),double(sin(d2r(theta_i)))];
sig_x = 8;
sig_y = 8;

Tmax = 1000;
dt = 0.05;
t = 0:dt:Tmax;

f = 1/(pi/0.9);
V = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
V(:,:,1) = 10.613;

n = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
n(:,:,1) = 0.1765;
m = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
m(:,:,1) = 0.0529;
h = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
h(:,:,1) = 0.5961;

g_ex = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
g_in = zeros(int16(size(B,1)),int16(size(B,2)),length(t));

R = 3;
% for theta_i=1:60:360
% for y_c_i=(int16(size(B,2)))/2:(int16(size(B,2)))/2
%     for x_c_i=(int16(size(B,1)))/2:(int16(size(B,1)))/2

% for x_c_i=64:64
%     for y_c_i=64:64

% for j=1:(length(t)-1)
for theta_i=1:359
    normal = [-double(cos(d2r(theta_i))),double(sin(d2r(theta_i)))];
    for j=1:1
        % PER NEURON
       for y_c_i=(int16(size(B,2)))/2:(int16(size(B,2)))/2
           for x_c_i=(int16(size(B,1)))/2:(int16(size(B,1)))/2
    %     for y_c_i=1:(int16(size(B,2)))
    %         for x_c_i=1:(int16(size(B,1)))
                W_ex = zeros(int16(size(B,1)),int16(size(B,2)));
                W_in = zeros(int16(size(B,1)),int16(size(B,2)));

                g_ex_sum = 0.0;
                g_in_sum = 0.0;

                % PER NEURON WINDOW
                for x=(x_c_i - R):(x_c_i + R)
                    if(x <= 0 || x > int16(size(B,1)))
                        continue
                    end
                    for y=(y_c_i - R):(y_c_i + R)
                        if(y <= 0 || y > int16(size(B,1)))
                            continue
                        end

                        x_theta_i = (x-x_c_i) * cos(d2r(theta_i)) + (y-y_c_i) * sin(d2r(theta_i));
                        y_theta_i = (y-y_c_i) * cos(d2r(theta_i)) - (x-x_c_i) * sin(d2r(theta_i));

                        v = [double(x_theta_i),double(x_theta_i)];
                        
                        % theta_i \in (45,225]
                        if(theta_i > 45 && theta_i <= 225)
                            if(dot(v,normal) <= 0)
                                W_ex(x,y) = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                                g_ex_sum = g_ex_sum + W_ex(x,y) * double(B(x,y)/255.0);
                            else
                                W_ex(x,y) = 0;
                            end

                            if(dot(v,normal) >= 0)
                                W_in(x,y) = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                                g_in_sum = g_in_sum + W_in(x,y) * double(B(x,y)/255.0);
                            else
                                W_in(x,y) = 0;
                            end
                        else
                            if(dot(v,normal) >= 0)
                                W_ex(x,y) = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                                g_ex_sum = g_ex_sum + W_ex(x,y) * double(B(x,y)/255.0);
                            else
                                W_ex(x,y) = 0;
                            end

                            if(dot(v,normal) <= 0)
                                W_in(x,y) = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                                g_in_sum = g_in_sum + W_in(x,y) * double(B(x,y)/255.0);
                            else
                                W_in(x,y) = 0;
                            end
                        end
                        D(x_c_i,y_c_i) = D(x_c_i,y_c_i) + W_ex(x,y) * (B(x,y)/255.0);
                    end
                end

                % COMPUTE PER NEURON POTENTIAL
                I_ex = g_ex(x_c_i, y_c_i,j) * (V(x_c_i, y_c_i,j) - E_ex);
                I_in = g_in(x_c_i, y_c_i,j) * (V(x_c_i, y_c_i,j) - E_in);
                I_K = G_K * (n(x_c_i, y_c_i, j)^4) * (V(x_c_i, y_c_i,j) - E_K);
                I_Na = G_Na * (m(x_c_i, y_c_i, j)^3) * h(x_c_i, y_c_i, j) * (V(x_c_i, y_c_i,j) - E_Na);

                k1V = (1.0/C) * (I_ex + I_in - I_K - I_Na - G_L * (V(x_c_i, y_c_i,j) - E_L));
                k1n = (n_inf(V(x_c_i, y_c_i,j)) - n(j))/tau_n(V(x_c_i, y_c_i,j));
                k1m = (m_inf(V(x_c_i, y_c_i,j)) - m(j))/tau_m(V(x_c_i, y_c_i,j));
                k1h = (h_inf(V(x_c_i, y_c_i,j)) - h(j))/tau_h(V(x_c_i, y_c_i,j));
                k1g_ex = -(1.0 / tau_ex) * g_ex(x_c_i, y_c_i,j) + g_ex_sum;
                k1g_in = -(1.0 / tau_in) * g_in(x_c_i, y_c_i,j) + g_in_sum;

                a2V1 = (V(x_c_i, y_c_i,j) + k1V * dt);
                a2n = (n(x_c_i, y_c_i, j) + k1n * dt);
                a2m = (m(x_c_i, y_c_i, j) + k1m * dt);
                a2h = (h(x_c_i, y_c_i, j) + k1h * dt);
                a2g_ex = (g_ex(x_c_i, y_c_i, j) + k1g_ex * dt);
                a2g_in = (g_in(x_c_i, y_c_i, j) + k1g_in * dt);

                I_ex_2 = a2g_ex * (a2V1 - E_ex);
                I_in_2 = a2g_in * (a2V1 - E_in);
                I_K_2 = G_K * (a2n^4) * (a2V1 - E_K);
                I_Na_2 = G_Na * (a2m^3) * a2h * (a2V1 - E_Na);

                k2V = (1.0/C) * (I_ex + I_in - I_K - I_Na - G_L * (a2V1 - E_L));
                k2n = (n_inf(a2V1) - a2n)/tau_n(a2V1);
                k2m = (m_inf(a2V1) - a2m)/tau_m(a2V1);
                k2h = (h_inf(a2V1) - a2h)/tau_h(a2V1);
                k2g_ex = -(1.0 / tau_ex) * a2g_ex + g_ex_sum;
                k2g_in = -(1.0 / tau_in) * a2g_in + g_in_sum;

                V(x_c_i, y_c_i,j+1) = V(x_c_i, y_c_i,j) + (k1V + k2V) * (dt/2.0);
                if(V(x_c_i, y_c_i, j+1) > 3e127)
                    V(x_c_i, y_c_i, j+1) = 3e127;
                end
                n(x_c_i, y_c_i,j+1) = n(x_c_i, y_c_i,j) + (k1n + k2n) * (dt/2.0);
                m(x_c_i, y_c_i,j+1) = m(x_c_i, y_c_i,j) + (k1m + k2m) * (dt/2.0);
                h(x_c_i, y_c_i,j+1) = h(x_c_i, y_c_i,j) + (k1h + k2h) * (dt/2.0);
                g_ex(x_c_i, y_c_i,j+1) = g_ex(x_c_i, y_c_i,j) + (k1g_ex + k2g_ex) * (dt/2.0);
                g_in(x_c_i, y_c_i,j+1) = g_in(x_c_i, y_c_i,j) + (k1g_in + k2g_in) * (dt/2.0);
            end
        end
    end
    imwrite(W_in, strcat('./filter_pictures/',config,'/in/', int2str(theta_i), '.png'));
    imwrite(W_ex, strcat('./filter_pictures/',config,'/ex/', int2str(theta_i), '.png'));
end
toc;
% end

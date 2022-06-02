clear;
close all;
Tmax_ani = 500;
tic;

A = imread("Project/Lenna128.png");
% A = imread("Project/lenna64.png");
% A = imread("Project/cityscapes-smol.jpeg");
% grayscales
% B = (0.3 * A(:,:,1) + 0.4 * A(:,:,2) + 0.1 * A(:,:,3));
% B = (0.2126 * A(:,:,1) + 0.7152 * A(:,:,2) + 0.0722 * A(:,:,3));
B = A(:,:,1);
% imshow(B);
% D = zeros(int16(size(B,1)),int16(size(B,2)));

% Initial Values / Constants
C = 1;
E_L = -55;
E_K = -90;
E_Na = 130;
G_L = 1;
G_K = 30;
G_Na = 130;

V_th = -65;
V_reset = -70;
E_ex = 0;
E_in = -70;
tau_ex = 0.8;
tau_in = 10;

I_app = 0.2;
alpha = 2;

% I = @(x,y) B(x + int16(size(B,2)/2) + 1,int16(size(B,1)) - (y + int16(size(B,1)/2) - 1));
I = @(x,y,a,i_app) a*(double(B(x,y))/255.0) + i_app;
d2r = @(d) d * (pi / 180);

m_inf=@(v) 1./(1+exp(-(v+40)/9));
h_inf=@(v) 1./(1+exp((v+50)/10));
n_inf=@(v) 1./(1+exp(-(v+53)/16));
tau_m=@(v) 0.3 + 0.00000*v;
tau_h=@(v) 1+11./(1+exp((v+50)/10));
tau_n=@(v) 1+6./(1+exp((v+53)/16));

% theta_i = 240;
theta_i = 0;
normal = [-double(cos(d2r(theta_i))),double(sin(d2r(theta_i)))];
sig_x = 8;
sig_y = 8;

Tmax = 1000;
dt = 0.1;
t = 0:dt:Tmax;

f = 1/(pi/0.9);
V = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
V(:,:,1) = E_L;

n = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
n(:,:,1) = 0.1765;
% n(:,:,1) = 1;
m = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
m(:,:,1) = 0.0529;
% m(:,:,1) = 0;
h = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
h(:,:,1) = 0.5961;
% h(:,:,1) = 0;

% W_ex = zeros(int16(size(B,1)),int16(size(B,2)));
% W_in = zeros(int16(size(B,1)),int16(size(B,2)));
g_ex = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
g_in = zeros(int16(size(B,1)),int16(size(B,2)),length(t));
g_ex_sum = zeros(int16(size(B,1)),int16(size(B,2)));
g_in_sum = zeros(int16(size(B,1)),int16(size(B,2)));

% gabor filter
R = 4;



R = 4;
x_dilation_rate = 1;
y_dilation_rate = 1;
for j=1:Tmax_ani
    % PER NEURON
   for y_c_i=60:60
       for x_c_i=13:13
%     for y_c_i=1:(int16(size(B,2)))
%         for x_c_i=1:(int16(size(B,1)))

            g_ex_sum = 0.0;
            g_in_sum = 0.0;
            c = 0;
            % PER NEURON WINDOW
            for x=(x_c_i - R*x_dilation_rate):x_dilation_rate:(x_c_i + R*x_dilation_rate)
                if(x <= 0 || x > int16(size(B,1)))
                    c = c + (2*R + 1);
                    continue
                end
                for y=(y_c_i - R*y_dilation_rate):y_dilation_rate:(y_c_i + R*y_dilation_rate)
                    c = c + 1;
                    if(y <= 0 || y > int16(size(B,1)))
                        continue
                    end

                    x_theta_i = (x-x_c_i) * cos(d2r(theta_i)) + (y-y_c_i) * sin(d2r(theta_i));
                    y_theta_i = (y-y_c_i) * cos(d2r(theta_i)) - (x-x_c_i) * sin(d2r(theta_i));

                    v = [double(x_theta_i),double(x_theta_i)];

                    % theta_i \in (45,225]
                    if(theta_i > 45 && theta_i <= 225)
                        if(dot(v,normal) <= 0)
                            W_ex = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                            g_ex_sum = g_ex_sum + W_ex * double(I(x,y,alpha,I_app));
                        else
                            W_ex = -1;
                            g_ex_sum = g_ex_sum + W_ex * double(I(x,y,alpha,I_app));
                        end

                        if(dot(v,normal) > 0)
                            W_in = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                            g_in_sum = g_in_sum + W_in * double(I(x,y,alpha,I_app));
                        else
                            W_in = -1;
                            g_in_sum = g_in_sum + W_in * double(I(x,y,alpha,I_app));
                        end
                    else
                        if(dot(v,normal) >= 0)
                            W_ex = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                            g_ex_sum = g_ex_sum + W_ex * double(I(x,y,alpha,I_app));
                        else
                            W_ex = -1;
                            g_ex_sum = g_ex_sum + W_ex * double(I(x,y,alpha,I_app));
                        end

                        if(dot(v,normal) < 0)
                            W_in = exp(-0.5 * (double((x_theta_i^2)/(sig_x^2)) + double((y_theta_i^2)/(sig_y^2)))) * cos(double(2*pi*f*x_theta_i));
                            g_in_sum = g_in_sum + W_in * double(I(x,y,alpha,I_app));
                        else
                            W_in = -1;
                            g_in_sum = g_in_sum + W_in * double(I(x,y,alpha,I_app));
                        end
                    end

%                     D(x_c_i,y_c_i) = D(x_c_i,y_c_i) + W_ex * (B(x,y)/255.0);
                end
            end
            
            g_in_sum = g_in_sum / c;
            g_ex_sum = g_ex_sum / c;
%             if(g_in_sum < 0)
%                 g_in_sum = 0;
%             end
%             if(g_ex_sum < 0)
%                 g_ex_sum = 0;
%             end

%             g_in_sum
%             g_ex_sum
            
            % COMPUTE PER NEURON POTENTIAL
            I_ex = g_ex(x_c_i, y_c_i,j) * (V(x_c_i, y_c_i,j) - E_ex);
            I_in = g_in(x_c_i, y_c_i,j) * (V(x_c_i, y_c_i,j) - E_in);
%             V(x_c_i, y_c_i,j)
%             I_ex = 0;
%             I_in = 0;
            I_K = G_K * (n(x_c_i, y_c_i, j)^4) * (V(x_c_i, y_c_i,j) - E_K);
            I_Na = G_Na * (m(x_c_i, y_c_i, j)^3) * h(x_c_i, y_c_i, j) * (V(x_c_i, y_c_i,j) - E_Na);
            
            k1V = (1.0/C) * (-I_K + -I_Na - G_L * (V(x_c_i, y_c_i,j) - E_L));
            k1n = (n_inf(V(x_c_i, y_c_i,j)) - n(x_c_i, y_c_i, j))/tau_n(V(x_c_i, y_c_i,j));
            k1m = (m_inf(V(x_c_i, y_c_i,j)) - m(x_c_i, y_c_i, j))/tau_m(V(x_c_i, y_c_i,j));
            k1h = (h_inf(V(x_c_i, y_c_i,j)) - h(x_c_i, y_c_i, j))/tau_h(V(x_c_i, y_c_i,j));
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
            
            k2V = (1.0/C) * (-I_ex_2 + -I_in_2 - I_K - I_Na - G_L * (a2V1 - E_L));
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
% figure
% subplot(1,2,1);
% imshow(W_in)
% title("Inhibitory Synapse Weight");
% subplot(1,2,2);
% imshow(W_ex)
% title("Exhibitory Synapse Weight");
% figure
% imshow(D)
toc;

tic;
% Tmax_ani = 500;
for j=1:Tmax_ani
    % temp = (V(:,:,j) + abs(min(V(:,:,j),[],'all')));
    U = (V(:,:,j) + abs(E_K)) ./ (abs(E_Na) + abs(E_K));
    imwrite(U, strcat('./project7/240test/', int2str(j), '.png'));
%     imwrite(V(:,:,j), strcat('./project6/60default-2/', int2str(j), '.png'));
    % imwrite(temp / max(temp,[],'all'), strcat('./120default/', int2str(j), '.png'));
end
toc;

% figure
% hold on
% Tmax_ani = 500;
% center_x = 62;
% center_y = 57;

center_x = 59;
center_y = 60;
l = 0;
r = 0;
% for center_x=63:65
%     for center_y=63:65
%         figure
%         hold on
        for x=(center_x-l):(center_x+r)
            for y=(center_y-l):(center_y+r)
                figure
        % for x=center:center
        %     for y=center:center
                v_t = zeros(Tmax_ani,1);
                n_t = zeros(Tmax_ani,1);
                m_t = zeros(Tmax_ani,1);
                h_t = zeros(Tmax_ani,1);
                g_ex_t = zeros(Tmax_ani,1);
                g_in_t = zeros(Tmax_ani,1);
                I_ex_t = zeros(Tmax_ani,1);
                I_in_t = zeros(Tmax_ani,1);
                I_Na_t = zeros(Tmax_ani,1);
                I_K_t = zeros(Tmax_ani,1);
                I_L_t = zeros(Tmax_ani,1);
                for i=1:Tmax_ani
                    v_t(i) = V(x,y,i);
                    n_t(i) = n(x,y,i);
                    m_t(i) = m(x,y,i);
                    h_t(i) = h(x,y,i);
                    g_ex_t(i) = g_ex(x,y,i);
                    g_in_t(i) = g_in(x,y,i);
                    
                    I_ex_t(i) = g_ex(x_c_i, y_c_i,i) * (V(x_c_i, y_c_i,i) - E_ex);
                    I_in_t(i) = g_in(x_c_i, y_c_i,i) * (V(x_c_i, y_c_i,i) - E_in);
                    I_K_t(i) = G_K * (n(x_c_i, y_c_i, i)^4) * (V(x_c_i, y_c_i,i) - E_K);
                    I_Na_t(i) = G_Na * (m(x_c_i, y_c_i, i)^3) * h(x_c_i, y_c_i, i) * (V(x_c_i, y_c_i,i) - E_Na);
                    I_L_t(i) = G_L * (V(x_c_i, y_c_i,i) - E_L);
                end
                hold on
                subplot(5,1,1)
                plot(v_t)
                title(strcat(int2str(x)," ",int2str(y)))
                subplot(5,1,2)
                hold on
                plot(n_t)
                plot(m_t)
                plot(h_t)
                legend("n","m","h")
                subplot(5,1,3)
                hold on
%                 plot(ones(Tmax_ani,1)*double(I(x,y,alpha,I_app)))
                plot(ones(Tmax_ani,1)*double(B(x,y) / 255.0))
                subplot(5,1,4)
                hold on
                plot((n_inf(v_t) - n_t)./tau_n(v_t))
                plot((m_inf(v_t) - m_t)./tau_m(v_t))
                plot((h_inf(v_t) - h_t)./tau_h(v_t))

                subplot(5,1,5)
                hold on
                plot(g_ex_t)
                plot(g_in_t)
                legend("G_{ex}","G_{in}")
                figure
                subplot(6,1,1)
                plot(I_ex_t);
                legend("I_{ex}")
                title(strcat(int2str(x)," ",int2str(y))," conductances")
                subplot(6,1,2)
                plot(I_in_t);
                legend("I_{in}")
                subplot(6,1,3)
                plot(I_K_t);
                legend("I_{K}")
                subplot(6,1,4)
                plot(I_Na_t);
                legend("I_{Na}")
                subplot(6,1,5)
                plot(I_L_t);
                legend("I_{L}")
                subplot(6,1,6)
                plot((1.0/C) * (I_ex_t + I_in_t - I_K_t - I_Na_t - I_L_t));
                legend("dv/dt")
            end
        end
%     end
% end
hold off
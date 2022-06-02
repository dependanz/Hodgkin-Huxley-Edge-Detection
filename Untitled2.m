% alpha_n=@(v) (0.01 * (10.0 + v))/(exp((10.0+v)/10.0) - 1.0);
% alpha_m=@(v) (0.1 * (25.0 + v))/(exp((25.0+v)/10.0) - 1.0);
% alpha_h=@(v) 0.07 * (exp(v/20.0));
% beta_n=@(v) 0.125 * exp(v/80.0);
% beta_m=@(v) 4.0 * exp(v/18.0);
% beta_h=@(v) 1.0 ./ (exp((30.0 + v) / 10.0) + 1.0);
% % 
% % m_inf = @(v) alpha_m(v) ./ (alpha_m(v) + beta_m(v));
% % h_inf = @(v) alpha_h(v) ./ (alpha_h(v) + beta_h(v));
% % n_inf = @(v) alpha_n(v) ./ (alpha_n(v) + beta_n(v));
% tau_m = @(v) 1.0 ./ (alpha_m(v) + beta_m(v));
% tau_h = @(v) 1.0 ./ (alpha_h(v) + beta_h(v));
% tau_n = @(v) 1.0 ./ (alpha_n(v) + beta_n(v));
close all;

% C = 1;
% E_L = 10.613;
% E_K = -12;
% E_Na = 115;
% G_L = 0.3;
% G_K = 36;
% G_Na = 120;
% 
% V_th = -65;
% V_reset = -70;
% E_ex = 0;
% E_in = -75;
% tau_ex = 3;
% tau_in = 10;
% A_ex = 0.028953;
% A_in = 0.014103;
% tau_ref = 10;

% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+62)/10));
% n_inf=@(v) 1./(1+exp(-(v+53)/16));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+62)/10));
% tau_n=@(v) 1+6./(1+exp((v+53)/16));

% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+62)/12));
% n_inf=@(v) 1./(1+exp(-(v+20)/16));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+62)/10));
% tau_n=@(v) 1+6./(1+exp((v+53)/16));

% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+62)/10));
% n_inf=@(v) 1./(1+exp(-(v+20)/15));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+62)/10));
% tau_n=@(v) 1+6./(1+exp((v+53)/16));

% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+62)/15));
% n_inf=@(v) 1./(1+exp(-(v+20)/30));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+62)/10));
% tau_n=@(v) 1+6./(1+exp((v+20)/30));

% m_inf=@(v) 1./(1+exp(-(v+40)/9));
% h_inf=@(v) 1./(1+exp((v+50)/10));
% n_inf=@(v) 1./(1+exp(-(v+35)/60));
% tau_m=@(v) 0.3 + 0.0*v;
% tau_h=@(v) 1+11./(1+exp((v+50)/10));
% tau_n=@(v) 1+6./(1+exp((v+53)/24));

alpha_n=@(v) (0.01 * (10.0 + v))/(exp((10.0+v)/10.0) - 1.0);
alpha_m=@(v) (0.1 * (25.0 + v))/(exp((25.0+v)/10.0) - 1.0);
alpha_h=@(v) 0.07 * (exp(v/20.0));
beta_n=@(v) 0.125 * exp(v/80.0);
beta_m=@(v) 4.0 * exp(v/18.0);
beta_h=@(v) 1.0 ./ (exp((30.0 + v) / 10.0) + 1.0);

m_inf = @(v) alpha_m(v) ./ (alpha_m(v) + beta_m(v));
h_inf = @(v) alpha_h(v) ./ (alpha_h(v) + beta_h(v));
n_inf = @(v) alpha_n(v) ./ (alpha_n(v) + beta_n(v));
tau_m = @(v) 1.0 ./ (alpha_m(v) + beta_m(v));
tau_h = @(v) 1.0 ./ (alpha_h(v) + beta_h(v));
tau_n = @(v) 1.0 ./ (alpha_n(v) + beta_n(v));

vv=-200:0.1:200;

figure
hold on
plot(vv,m_inf(vv),'b','linewidth',2);
plot(vv,h_inf(vv),'r','linewidth',2);
plot(vv,n_inf(vv),'g','linewidth',2);
axis([-200 200 -0.1 1.1]);
set(gca,'fontsize',20);
xlabel('V');
legend('m_{\infty}','h_{\infty}','n_{\infty}');
title("activation/inactivation functions")

figure
hold on
plot(vv,tau_m(vv),'b','linewidth',2);
plot(vv,tau_h(vv),'r','linewidth',2);
plot(vv,tau_n(vv),'g','linewidth',2);
axis([-200 200 -0.1 12]);
set(gca,'fontsize',20);
xlabel('V');
ylabel('ms');
legend('\tau_{m}','\tau_{h}','\tau_{n}');
title("Voltage-dependent time constants")
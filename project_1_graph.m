close all;
% figure
% hold on
Tmax_ani = 500;
% center_x = 62;
% center_y = 57;

center_x = 59;
center_y = 60;
l = 3;
r = 3;
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
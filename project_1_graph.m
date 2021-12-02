close all;
% figure
% hold on
Tmax_ani = 500;
center_x = 64;
center_y = 64;
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
                for i=1:Tmax_ani
                    v_t(i) = V(x,y,i);
                    n_t(i) = n(x,y,i);
                    m_t(i) = m(x,y,i);
                    h_t(i) = h(x,y,i);
                end
                hold on
                subplot(3,1,1)
                plot(v_t)
                subplot(3,1,2)
                hold on
                plot(n_t)
                plot(m_t)
                plot(h_t)
                legend("n","m","h")
                subplot(3,1,3)
                hold on
                plot(ones(Tmax_ani,1)*double(B(x,y))/255.0)
                hold off
            end
        end
%     end
% end
hold off
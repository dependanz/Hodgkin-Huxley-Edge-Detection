close all;
% figure
% hold on
Tmax_ani = 500;
center_x = 64;
center_y = 4;
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
                for i=1:Tmax_ani
                    v_t(i) = V(x,y,i);
                end
                plot(v_t)
            end
        end
%     end
% end
hold off
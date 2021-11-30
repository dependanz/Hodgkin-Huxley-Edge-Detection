close all;
% figure
% hold on
center_x = 64;
center_y = 64;
l = 0;
r = 0;
for center_x=50:2:80
    for center_y=50:2:80
        figure
        for x=(center_x-l):(center_x+r)
            for y=(center_y-l):(center_y+r)
        % for x=center:center
        %     for y=center:center
                v_t = zeros(1000,1);
                for i=1:1000
                    v_t(i) = V(x,y,i);
                end
                plot(v_t)
            end
        end
    end
end
hold off
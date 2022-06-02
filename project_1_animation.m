tic;
Tmax_ani = 1000;
for j=1:Tmax_ani
    % temp = (V(:,:,j) + abs(min(V(:,:,j),[],'all')));
    U = (V(:,:,j) + abs(E_K)) ./ (abs(E_Na) + abs(E_K));
%     imwrite(U, strcat('./project7/240again/', int2str(j), '.png'));
    imwrite(U, strcat('./presentation/240/', int2str(j), '.png'));
%     imwrite(V(:,:,j), strcat('./project6/60default-2/', int2str(j), '.png'));
    % imwrite(temp / max(temp,[],'all'), strcat('./120default/', int2str(j), '.png'));
end
toc;
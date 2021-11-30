for j=1:1000
    % temp = (V(:,:,j) + abs(min(V(:,:,j),[],'all')));
    imwrite(V(:,:,j)/E_Na, strcat('./project2/60default/', int2str(j), '.png'));
    % imwrite(temp / max(temp,[],'all'), strcat('./120default/', int2str(j), '.png')); 
end
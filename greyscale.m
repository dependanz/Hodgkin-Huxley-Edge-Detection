A = imread("Project/Lenna128.png");
% A = imread("Project/lenna64.png");
% A = imread("Project/cityscapes-smol.jpeg");
% grayscales
% B = (0.3 * A(:,:,1) + 0.4 * A(:,:,2) + 0.1 * A(:,:,3));
B = (0.2126 * A(:,:,1) + 0.7152 * A(:,:,2) + 0.0722 * A(:,:,3));
imshow(B);
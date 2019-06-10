% clean = rgb2gray(imread('baby_GT.bmp'));
% maxval = max(max(clean));
% 
% 
% clean_scaled = im2double(clean);
% 
% noisy = im2double(imnoise(clean,'poisson'));
% 
% peak = max(max(clean_scaled));
% 
% save poisson_baby.mat



%Generate noisy and blurry image
clean =double(rgb2gray(imread('woman_GT.bmp')));
maxval=max(clean(:));
peak    = 1;
%kernel=fspecial('gaussian',25,1.6);
clean_scaled=clean*peak/maxval;
clean_scaled(clean_scaled == 0) = min(min(clean_scaled(clean_scaled > 0)));
noisy = knuth_poissrnd(clean_scaled);

save poisson_woman.mat
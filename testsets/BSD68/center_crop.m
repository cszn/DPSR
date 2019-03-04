function [iout] = center_crop(im,a,b)

[w,h,~] = size(im);
c1 = w-a-round((w-a)/2);
c2 = h-b-round((h-b)/2);
iout = im(c1+1:c1+a,c2+1:c2+b,:);

end


function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = center_crop(imgs,sz(1),sz(2));
    %imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    
    imgs = center_crop(imgs,sz(1),sz(2));
    %imgs = imgs(1:sz(1), 1:sz(2),:);
end


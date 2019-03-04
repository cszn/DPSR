


input_folder    = 'Set5';
GT = 'GT';

original_ext    = {'*.png', '*.bmp', '*.jpg'};

images          = [];

for i = 1:length(original_ext)
    images = [images; dir(fullfile(input_folder, GT, original_ext{i}))];
end


for sf = [2, 3, 4]
    
    output_folder = fullfile(input_folder,['x',num2str(sf, '%01d')]);
    
    if isdir(output_folder) == 0
        mkdir(output_folder);
    end
    
    for i = 1:numel(images)
        
        [~, name, exte] = fileparts(images(i).name);
        I = imread(fullfile(input_folder,GT,images(i).name)) ;
        I = modcrop(I, sf);
        I = imresize(I,1/sf, 'bicubic');
        imshow(I)
        
        imwrite(I, fullfile(output_folder, images(i).name));
    end
end









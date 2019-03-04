


input_folder    = 'GT';

kerneltype = {'g','m','d'};

original_ext    = {'*.png'};
target_ext      = '.png';

images          = [];

for ii = 1:length(original_ext)
    images = [images; dir(fullfile(input_folder, original_ext{ii}))];
end

for sf = 2:4
    for kn = 1:3
        output_folder   = ['x',num2str(sf),'_',kerneltype{kn}];
        
        if isdir(output_folder) == 0
            mkdir(output_folder);
        end
        disp([sf,kn]);
        for ii = 1:numel(images)
            [~, name, exte] = fileparts(images(ii).name);
            I   =   imread(fullfile(input_folder,images(ii).name) ) ;
            % I = modcrop(I,lcm(16,3));
            I = imresize(I, 1/sf, 'bicubic');
            currentkernels = dir(fullfile('the_kernels',[kerneltype{kn},'*.mat']));
            for currentkn = 1:length(currentkernels)
                [~, kname, kexte] = fileparts(currentkernels(currentkn).name);
                ckernel = load(fullfile('the_kernels',currentkernels(currentkn).name));
                kernel = ckernel.kernel;
                LR = imfilter(im2double(I), kernel, 'circular', 'conv');
                imwrite(LR, fullfile(output_folder,[name,'_',kname,target_ext]));
                save(fullfile(output_folder,[name,'_',kname,'.mat']),'kernel')
            end
        end
    end
end








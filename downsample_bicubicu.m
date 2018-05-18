close all;
clear all;

%% 

dataDir = './';%fullfile('data', '291');
mkdir('LR_bicubicu');

count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.jpg'))];
%f_lst = [f_lst; dir(fullfile(dataDir, '*.bmp'))];


for f_iter = 1:numel(f_lst)
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    img_size = size(img_raw);
    
    img_2 = imresize(img_raw,1/2,'bicubic');
    img_3 = imresize(img_raw,1/3,'bicubic');
    img_4 = imresize(img_raw,1/4,'bicubic');   
    
    file_name = regexp(f_info.name, '.jpg', 'split');
    file_name = file_name{1};
    
    display(file_name)
    img_name = sprintf('LR_bicubicu/%s',file_name);
    imwrite(img_2, sprintf('%s_LRx2.png', img_name));
    imwrite(img_3, sprintf('%s_LRx3.png', img_name));
    imwrite(img_4, sprintf('%s_LRx4.png', img_name));;
end

function [img_cube,gt_cube] = load_samples(imgs_list,gts_list,samples_idx_struct)

% collect the patches of images and groundtruth

imgs_no = length(imgs_list);

% n_samples = size(samples_idx_struct.idxs,1);


switch imgs_list{1}(end - 2:end)
    
    case 'dcm'
        
        img = load_PA_data(imgs_list{1});

        gt_img = load_PA_data(gts_list{1});

    case 'jpg'
    
        img = imread(imgs_list{1});

        gt_img = imread(gts_list{1});
        
        
    case 'tif'

        img = load_PA_data(imgs_list{1});
        
        gt_img = load_PA_data(gts_list{1});
        
        
    otherwise
        
end
        

cube_size = samples_idx_struct.patch_size;

img_cube = repmat(img(1),length(samples_idx_struct.idxs),prod(cube_size));

gt_cube =  repmat(gt_img(1),length(samples_idx_struct.idxs),prod(cube_size));

for i_img = 1 : imgs_no,
        
    img_fn = imgs_list{i_img};
    
    img = load_data(img_fn);
    
    gt_fn = gts_list{i_img};
    
    gt_img = load_data(gt_fn);

    
    if(length(cube_size) == 2)
       
        img = img(:,:,2);
        
        gt_img = gt_img(:,:,1);
        
    end
    
    
    if(size(gt_img,3) > 1)
        
        [img,gt_img] = crop_align(img,gt_img);
        
%         sz_img = size(img);
%         
%         sz_gt = size(gt_img);
%         
%         sz_img = min([sz_img;sz_gt]);
%         
%         img = img(1:sz_img(1),1:sz_img(2),1:sz_img(3));
%         
%         gt_img = gt_img(1:sz_img(1),1:sz_img(2),1:sz_img(3));
%         
    end
    
    idx_img = samples_idx_struct.idxs(samples_idx_struct.img_idxs == i_img,:);

    img_cube(samples_idx_struct.img_idxs == i_img,:) = collect_patches(img,idx_img,cube_size);
    
    gt_cube(samples_idx_struct.img_idxs == i_img,:) = collect_patches(gt_img,idx_img,cube_size);

    
    
    
    
    
end






function [img_cube,gt_cube] = load_samples_3D(imgs_list,gts_list,samples_idx_struct)

% collect the patches and cuebes of images and groundtruth


imgs_no = length(imgs_list);

n_samples = size(samples_idx_struct.idxs,1);

p = samples_idx_struct.p;

ftrs_type = p.ftrs;

cube_size = samples_idx_struct.patch_size;

gt_cube =  repmat(length(samples_idx_struct.idxs),prod(cube_size));


switch ftrs_type
    
    case 'Gabor'
        
        img_cube = zeros(n_samples,84);
        
    case 'Coye'
        
        img_cube = zeros(n_samples,9);
        
    otherwise
        
end
   
for i_img = 1 : imgs_no
        
    img_fn = imgs_list{i_img};
    
    img = load_data(img_fn);
    
%     Adhoc_Retinal(img);
    
    gt_fn = gts_list{i_img};
    
    gt_img = load_data(gt_fn);

    gt_img = gt_img > 0;
    
    [img,gt_img] = crop_align(img,gt_img);
    
%     if(length(cube_size) == 2)
%        
%         img = img(:,:,2);
%         
%         gt_img = gt_img(:,:,1);
%         
%     end

    idx_img = samples_idx_struct.idxs(samples_idx_struct.img_idxs == i_img,:);
    
    n_ftrs = 1;
    
    switch ftrs_type
        
        case 'RGB'
            
            for ib = 1 : size(img,3)
                
                tmp_img_cube = collect_patches(img(:,:,ib),idx_img,cube_size);
                
                n_ftrs_tmp = size(tmp_img_cube,2);
                
                img_cube(samples_idx_struct.img_idxs == i_img,n_ftrs : (n_ftrs + n_ftrs_tmp - 1)) = tmp_img_cube;
                
                n_ftrs = n_ftrs + n_ftrs_tmp;
                
            end
            
        case 'Gabor'
            
            [x,y,z] = ind2sub(size(img),idx_img); 
            
            xy = sub2ind(size(img(:,:,1)),x,y);
            
            zs = unique(z);
            
            tmp_img_cube = zeros(length(idx_img),84);
            
            for zi = 1 : length(zs)
                
                features = Gabor_Filter(img(:,:,zs(zi)));
                
                tmp_img_cube(z == zs(zi),:) = features(xy(z == zs(zi)),:);
                
            end
            
            img_cube(samples_idx_struct.img_idxs == i_img,:) = tmp_img_cube;
            
       case 'Coye'     
           
            [x,y,z] = ind2sub(size(img),idx_img); 
            
            xy = sub2ind(size(img(:,:,1)),x,y);
            
            zs = unique(z);
            
            tmp_img_cube = zeros(length(idx_img),9);
            
            for zi = 1 : length(zs)
                
                features = CoyeFilter_v2(img(:,:,zs(zi)));
                
                tmp_img_cube(z == zs(zi),:) = collect_patches(features,xy(z == zs(zi)),[3,3]);
                
            end
            
            img_cube(samples_idx_struct.img_idxs == i_img,:) = tmp_img_cube;
           
        otherwise
            
    end
       
    gt_cube(samples_idx_struct.img_idxs == i_img,:) = collect_patches(gt_img,idx_img,cube_size);
    
end
function [score_img,tbc_imgs,img_ftrs,idx_img] = RF_predict_retinal_image(img,mask_img,forest,p)

% predict both the estimation score and the tree based code

patch_size = p.patch_size;

border_mask_factor = ceil(patch_size / 2);

borderMask = zeros(size(img(:,:,1)),'logical');

if(p.dimension ==2)
    
    borderMask( 1:border_mask_factor(1), : ) = 1;
    borderMask( :, 1:border_mask_factor(2) ) = 1;
    borderMask( :, (end-border_mask_factor(2)):end ) = 1;
    borderMask( (end-border_mask_factor(1)):end, : ) = 1;
    
else
    borderMask( 1:border_mask_factor(1), : ,:) = 1;
    borderMask( :, 1:border_mask_factor(2) ,:) = 1;
    borderMask( :, :,1:border_mask_factor(3)) = 1;
    borderMask( :, :,(end-border_mask_factor(3)):end ) = 1;
    borderMask( :,(end-border_mask_factor(2)):end, : ) = 1;
    borderMask( (end-border_mask_factor(1)):end,:,: ) = 1;
end

borderMask = borderMask == 0;

borderMask = borderMask & mask_img;

idx_img = find(borderMask);

n_ftrs = 1;

img_cube = zeros(length(idx_img),prod(patch_size) * 3);

for ib = 1 : size(img,3)
    
    tmp_img_cube = collect_patches(img(:,:,ib),idx_img,patch_size);
    
    n_ftrs_tmp = size(tmp_img_cube,2);
    
    img_cube(:,n_ftrs : (n_ftrs + n_ftrs_tmp - 1)) = tmp_img_cube;
    
    n_ftrs = n_ftrs + n_ftrs_tmp;
    
end

clear tmp_img_cube;


img_Gabor = Gabor_Filter(img(:,:,2));

img_Gabor = img_Gabor(idx_img,:);

img_Coye = CoyeFilter(img);

img_Coye = collect_patches(img_Coye,idx_img,[3,3]);




img_ftrs = single([img_cube,img_Gabor,img_Coye]); 

[~,ps,tcs] = forestApply_leaves(img_ftrs,forest);

score_img = zeros(size(img(:,:,1)));

score_img(idx_img) = ps(:,2);

tbc_imgs = zeros([size(img(:,:,1)) length(forest)],'uint8');


for t = 1 : length(forest)
   
    tmp_tbc = zeros(size(img(:,:,1)),'uint8');
    
    tmp_tbc(idx_img) = uint8(tcs(:,t));
    
    tbc_imgs(:,:,t) = tmp_tbc;
    

end


     
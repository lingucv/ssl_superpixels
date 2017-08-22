function samples_return = get_samples_masks(p,imgs_list,gts_list,masks_list)
% 


pos_no = p.pos_sample_no(1);
neg_no = p.neg_sample_no(1);
imgs_no = length(imgs_list);


samples.idx_pos = zeros(pos_no,1,'uint32'); % linear index
samples.idx_neg = zeros(neg_no,1,'uint32');

samples.gt_pos = zeros(pos_no,1,'logical');
samples.gt_neg = zeros(neg_no,1,'logical');

samples.idx_img_pos= zeros(pos_no,1,'uint32'); 
samples.idx_img_neg= zeros(neg_no,1,'uint32');

n_pos_samples_per_image = zeros(imgs_no,1,'uint32'); 
n_neg_samples_per_image = zeros(imgs_no,1,'uint32'); 


patch_size = p.patch_size;

border_mask_factor = ceil(patch_size / 2);


% get samples
pos_samples_collected_no = 0;

neg_samples_collected_no = 0;

for i_img = 1 : imgs_no
    
    gt_fn = gts_list{i_img};
    
    switch gt_fn(end - 2:end)
        
        case 'tif'
            
            gt_img = load_PA_data(gt_fn);
            
        case 'png'
            
            gt_img = imread(gt_fn);
            
        case 'gif'
            
            gt_img = imread(gt_fn);
            
        case 'ppm'
            
            gt_img = imread(gt_fn);
            
        otherwise
            
    end
    
    % change the groundtruth data into the binary format for the sake of
    % memory issue
    
    max_gt = max(gt_img(:));
    
    gt_img = gt_img > (max_gt / 2);
    
    if (i_img < imgs_no)
        
        N_pos = floor(double(pos_no) / single(imgs_no));
        
        N_neg = floor(double(neg_no) / single(imgs_no));
        
    else
        
        N_pos = pos_no - pos_samples_collected_no;
        
        N_neg = neg_no - neg_samples_collected_no;
        
    end
    
    
    borderMask = zeros(size(gt_img),'logical');
    
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
    %
    borderMask = borderMask == 0;
    
    if(~isempty(masks_list))
        
        mask_fn = masks_list{i_img};
        
        switch mask_fn(end - 2:end)
            
            case 'tif'
                
                mask_img = load_PA_data(mask_fn);
                
            case 'png'
                
                mask_img = imread(mask_fn);
                
            case 'gif'
                
                mask_img = imread(mask_fn);
                
            otherwise
                
        end
        
        mask_img = mask_img > 0;
                
        borderMask = borderMask & mask_img;
                
    end
     
     if(isfield(p,'pos_sample_strategy'))
         
         switch p.pos_sample_strategy
             
             case 'centreline'
                 
                 if(p.dimension > 2)
                 
                    sample_region = Skeleton3D(gt_img);
                 
                 else
                 
                    sample_region = bwmorph(gt_img,'skel',Inf);
                    
                 end
                 
             case 'away_from_boundary+centreline'
                 
                 sample_region = bwdist(~gt_img);
                 
                 sample_region = sample_region > p.pos_sample_distance;
                 
                 sample_region = bwmorph(sample_region,'skel',Inf);
                 
                 
             case 'away_from_boundary'
                 
                 sample_region = bwdist(~gt_img);
                 
                 sample_region = sample_region > p.pos_sample_distance;
                 
             otherwise
                 
         end
         
         pos_idx = sample_image_efficient(sample_region & borderMask,N_pos);
                  
     else
         
         pos_idx = sample_image_efficient(gt_img & borderMask,N_pos);
     
     end
  
     if(isfield(p,'neg_sample_distance'))
         
         sample_region = bwdist(gt_img);
         
         sample_region = sample_region > p.neg_sample_distance;

         neg_idx = sample_image_efficient(sample_region & borderMask,N_neg);
         
     else
         
         neg_idx = sample_image_efficient((~gt_img) & borderMask,N_neg);
         
     end
     
    samples.gt_pos(pos_samples_collected_no+(1:N_pos)) = gt_img(pos_idx);
    
    samples.gt_neg(neg_samples_collected_no+(1:N_neg)) = gt_img(neg_idx);
    
    samples.idx_pos(pos_samples_collected_no+(1:N_pos)) = uint32(pos_idx);% idxs wrt to size img
    
    samples.idx_neg(neg_samples_collected_no+(1:N_neg)) = uint32(neg_idx);% idxs wrt to size img
    
    
    samples.idx_img_pos(pos_samples_collected_no+(1:N_pos)) = i_img;
    
    samples.idx_img_neg(neg_samples_collected_no+(1:N_neg)) = i_img;
    
    pos_samples_collected_no = pos_samples_collected_no + N_pos;
    neg_samples_collected_no = neg_samples_collected_no + N_neg;
    
    n_pos_samples_per_image(i_img) = N_pos;
    n_neg_samples_per_image(i_img) = N_neg;
    
end

if(pos_no ~= pos_samples_collected_no  || neg_no~=neg_samples_collected_no)
    
   warning('number of collected samples differnt from number of required samples, this might create problems in training')

end

samples_return.idxs = [samples.idx_pos;samples.idx_neg];
samples_return.img_idxs = [samples.idx_img_pos;samples.idx_img_neg];

max_gt = max(samples.gt_pos);
samples.gt_pos = ones(size(samples.gt_pos),'single');
samples.gt_neg = 2*single(samples.gt_neg == max_gt)-1;

samples_return.labels = [samples.gt_pos;samples.gt_neg];
samples_return.n_pos_samples_per_image = n_pos_samples_per_image;
samples_return.n_neg_samples_per_image = n_neg_samples_per_image;


samples_return.patch_size = p.patch_size;

samples_return.p = p;

end

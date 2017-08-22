function samples_return = get_samples_masks_v2(p,imgs_list,gts_list,masks_list)
% 

% % use fixed dilat_factor instead
% dilat_factor = single(6); % this should be half_filters_size (for regression)

% 
% if(strcmp(p.loss_type,'exp') || strcmp(p.loss_type,'log') )
%     dilat_factor = single(0); % for classification positives only on centerlines
% end


pos_no = p.pos_sample_no(1);
neg_no = p.neg_sample_no(1);
imgs_no = length(imgs_list);
% use_scales_neg_mask = p.all_scales(p.all_scales~=p.train_scales(scale));
%use_scales_neg_mask = [];


samples.idx_pos = zeros(pos_no,1,'uint32'); % linear index
samples.idx_neg = zeros(neg_no,1,'uint32');
% samples.scale_pos_idx = zeros(pos_no,1,'uint8');
% samples.scale_neg_idx = zeros(neg_no,1,'uint8');  % index of scale wrt p.all_scales
% samples.idx_scale_sep_pos = zeros(pos_no,1,'uint8');
% samples.idx_scale_sep_neg = zeros(neg_no,1,'uint8'); 


samples.gt_pos = zeros(pos_no,1,'logical');
samples.gt_neg = zeros(neg_no,1,'logical');

samples.idx_img_pos= zeros(pos_no,1,'uint32'); 
samples.idx_img_neg= zeros(neg_no,1,'uint32');

n_pos_samples_per_image = zeros(imgs_no,1,'uint32'); 
n_neg_samples_per_image = zeros(imgs_no,1,'uint32'); 

% [scales_for_rescale,~,idx_scales_for_rescale] = intersect(p.predict_scales{scale},p.all_scales);
% scales_for_rescale = uint8(scales_for_rescale);

% max_filter_size = single(max(p.filters_size{scale})*max(scales_for_rescale))/ p.train_scales(scale);

max_filter_size = 10;

% border_mask_factor = round(max_filter_size/2 + single(max(p.max_cont_step)));
% 
% border_mask_factor = border_mask_factor(1);

patch_size = p.patch_size;

border_mask_factor = ceil(patch_size / 2);


% get samples
pos_samples_collected_no = 0;

neg_samples_collected_no = 0;

for i_img = 1 : imgs_no
    
    %     img_fn = imgs_list{i_img};
    %
    %     img = load_PA_data(img_fn);
    
    gt_fn = gts_list{i_img};
    
    switch gt_fn(end - 2:end)
        
        case 'tif'
            
            gt_img = double(load_PA_data(gt_fn));
           
            gt_img(gt_img < 1) = -1;
            
        case 'jpg'    
            
            gt_img = double(imread(gt_fn));
            
            gt_img(gt_img < 1) = -1;
            
        case 'png'
            
            gt_img = double(imread(gt_fn));
            
            gt_img(gt_img < 1) = -1;
            
        case 'gif'
            
            gt_img = imread(gt_fn);
            
            gt_img(gt_img < 1) = -1;
            
        case 'ppm'
            
            gt_img = double(imread(gt_fn));
            
            gt_img(gt_img < 1) = -1;
            
        case 'mat'
            
            gt_data = load(gt_fn);
            
            gt_img = zeros(size(gt_data.seg));
            
            for il = 1 : length(gt_data.allshadow)
                
                gt_img(gt_data.seg == gt_data.allshadow(il)) = 1;
                
            end
            
            
            for il = 1 : length(gt_data.allnonshadow)
                
                gt_img(gt_data.seg == gt_data.allnonshadow(il)) = -1;
                
            end
            
        otherwise
            
    end
    
    % change the groundtruth data into the binary format for the sake of
    % memory issue
    
%     max_gt = max(gt_img(:));
%     
    gt_img(gt_img > 0) = 1;
    
    if (i_img < imgs_no)
        
        N_pos = floor(double(pos_no) / single(imgs_no));
        
        N_neg = floor(double(neg_no) / single(imgs_no));
        
    else
        
        N_pos = pos_no - pos_samples_collected_no;
        
        N_neg = neg_no - neg_samples_collected_no;
        
    end
    
    
    %     [gt_max,scale_max] = max(gt_img(:,:,idx_scales_for_rescale),[],p.dimension+1);
    
    %     scale_max = uint8(scale_max);
    %     gt_dil = imdilate(gt_max==max(gt_max(:)),ones((2*dilat_factor+1)*ones(1,p.dimension)));
    %
    %     gt_zero_scale = zeros(size(gt_dil(:)));
    %     gt_zero_scale(gt_dil==0)=1;
    %     gt_zero_scale = uint8(gt_zero_scale);
    %     %clear gt_dil gt_max
    %
    
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
                 
                 sample_region = bwdist(gt_img < 0);
                
                 sample_region = sample_region > p.pos_sample_distance;
                 
             case 'none'
                 
                 sample_region = gt_img > 0;
                 
             otherwise
                 
         end
         
         pos_idx = sample_image_efficient(sample_region & borderMask,N_pos);
                  
     else
         
         pos_idx = sample_image_efficient(gt_img & borderMask,N_pos);
     
     end
     
   %  cl_gt = Skeleton3D(gt_img > 0);
     
   
   
     if(isfield(p,'neg_sample_distance'))
         
         sample_region = bwdist(gt_img > 0);
         
         sample_region = sample_region > p.neg_sample_distance;

         neg_idx = sample_image_efficient(sample_region & borderMask,N_neg);
         
     else
         
         neg_idx = sample_image_efficient((gt_img < 0) & borderMask,N_neg);
         
     end
     

%     
%     roi_mask = gt_img & borderMask;
%     
%     pos_idx = uint32(find(roi_mask(:)));
%     
%     pos_idx = data_sample(pos_idx,N_pos);
%     
%     for i_scale = 1:length(scales_for_rescale),
%         pos_idx{i_scale} = uint32(find((not_gt_zero_scale) & borderMask  & (scale_max(:) == i_scale)));
%     end
    
%     roi_mask = (~gt_img) & borderMask;
%     
%     neg_idx = uint32(find(roi_mask(:)));
%     
%     neg_idx = data_sample(neg_idx,N_neg);
    
        
% 
%         negative samples also at other scales
%          neg_ok = gt_zero_scale & borderMask;
%          neg_idx = uint32(find(neg_ok));
%          neg_idx_mask = [];
     


     

%     if(~isempty(use_scales_neg_mask))
% 
%         radial_gt = single(double(radial_gts{i_img}));
% 
%         mask_neg_scales = zeros(size(radial_gt),'uint8');
%         for i_s = 1:length(use_scales_neg_mask)
%             mask_neg_scales = mask_neg_scales | abs(radial_gt-use_scales_neg_mask(i_s))<=p.scale_toll(scale);
%         end
%         mask_neg_scales = imdilate(mask_neg_scales >0 ,ones((2*dilat_factor+1)*ones(1,p.dimension)));
%         mask_train_scale = abs(radial_gt-p.train_scales(scale))<=p.scale_toll(scale);
%         mask_train_scale = imdilate(mask_train_scale >0 ,ones((2*dilat_factor+1)*ones(1,p.dimension)));
% 
%         mask_train_scale = ~mask_train_scale(:) & borderMask;
%     
%         neg_idx_mask = uint32(find(mask_train_scale & mask_neg_scales(:))); % take neg samples from neighb. of scales i don't want to predict
%     end

% 
%         %%%%
%         % take randomly from all scales
%        pos_idx =  cell2mat(pos_idx);
%        Wsample = ones(1,length(pos_idx),'single');
%        Wsample = Wsample/sum(Wsample);
%         if(~isempty(pos_idx))
%               pos_swapping_scale = uint32(randsample(length(pos_idx),N_pos,true,Wsample));    
%         
%         else
%              warning('can not find positive samples for image %i ',i_img)
%             pos_swapping_scale = [];
%         end
        
        
%             
%         resize_scales = scale_max(pos_idx(pos_swapping_scale)); 
%         pos_samples_coord_scale = pos_idx(pos_swapping_scale);
%         %%%%%
%         
%         resize_scales_idx = resize_scales;
%         resize_scales = resize_scales + uint8(min(idx_scales_for_rescale)-1);
        
       
%     if(~isempty(neg_idx) && isempty(neg_idx_mask))
%         
%         WnegSample = ones(1,length(neg_idx),'single');
%          neg_swapping = uint32(randsample(length(neg_idx),N_neg,true,WnegSample));
%          neg_samples_coord = neg_idx(neg_swapping);
%          
%     elseif(~isempty(neg_idx) && ~isempty(neg_idx_mask)) % take portion of negative samples from other scales
%          
%         WnegSample1 = ones(1,length(neg_idx),'single');
%         WnegSample2 = ones(1,length(neg_idx_mask),'single');
%             
%         N_neg_1 = floor(4*N_neg/5);
%         N_neg_2 = max(N_neg - N_neg_1,1);
%         
%         neg_swapping_1 = uint32(randsample(length(neg_idx),N_neg_1,true,WnegSample1));
%         neg_swapping_2 = uint32(randsample(length(neg_idx_mask),N_neg_2,true,WnegSample2));
%         
%         neg_samples_coord = [neg_idx(neg_swapping_1);neg_idx_mask(neg_swapping_2)];
%      else
%         %neg_swapping = [];
%         N_neg = 0;
%         neg_samples_coord = [];
%     end
        
%    
%   
%     resize_scales_neg = uint8(find(p.all_scales == p.train_scales(scale))*ones(length(neg_samples_coord),1));
%     resize_scales_neg_idx = uint8(find(p.predict_scales{scale} == p.train_scales(scale))*ones(length(neg_samples_coord),1));
%     
%     
%        [r_neg,c_neg] = ind2sub(size(img),neg_samples_coord);%
%         
%         idx_neg = sub2ind(size(gt_img),r_neg,c_neg,uint32(resize_scales_neg));% idxs wrt to size gt
%         
% 
% 
%         [r_pos_scale,c_pos_scale] = ind2sub(size(img),pos_samples_coord_scale);
%         idx_pos = sub2ind(size(gt_img),r_pos_scale,c_pos_scale,uint32(resize_scales)); % idxs wrt to size gt
% 


    % get gt

    samples.gt_pos(pos_samples_collected_no+(1:N_pos)) = gt_img(pos_idx);
    
    samples.gt_neg(neg_samples_collected_no+(1:N_neg)) = gt_img(neg_idx) > 0;
    
    samples.idx_pos(pos_samples_collected_no+(1:N_pos)) = uint32(pos_idx);% idxs wrt to size img
    
    samples.idx_neg(neg_samples_collected_no+(1:N_neg)) = uint32(neg_idx);% idxs wrt to size img
    
    
    samples.idx_img_pos(pos_samples_collected_no+(1:N_pos)) = i_img;
    
    samples.idx_img_neg(neg_samples_collected_no+(1:N_neg)) = i_img;
    
%     samples.scale_pos_idx(pos_samples_collected_no+(1:N_pos)) = resize_scales;
%     samples.scale_neg_idx(neg_samples_collected_no+(1:N_neg)) = resize_scales_neg; % this is always idx train scale (?)
%     samples.idx_scale_sep_pos(pos_samples_collected_no+(1:N_pos)) = resize_scales_idx;
%     samples.idx_scale_sep_neg(neg_samples_collected_no+(1:N_neg)) = resize_scales_neg_idx;
    
    
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

if(strcmp(p.loss_type,'exp') || strcmp(p.loss_type,'log') )
    max_gt = max(samples.gt_pos);
    samples.gt_pos = ones(size(samples.gt_pos),'single');
    samples.gt_neg = 2*single(samples.gt_neg == max_gt)-1;
    
end

samples_return.labels = [samples.gt_pos;samples.gt_neg];
% samples_return.idx_scale_sep_cell = [samples.idx_scale_sep_pos;samples.idx_scale_sep_neg];
samples_return.n_pos_samples_per_image = n_pos_samples_per_image;
samples_return.n_neg_samples_per_image = n_neg_samples_per_image;


samples_return.patch_size = p.patch_size;

samples_return.p = p;

end

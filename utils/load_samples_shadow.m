function [trn_X,trn_Y] = load_samples_shadow(imgs_list,gts_list,samples_idx_struct)

% collect the patches of images and groundtruth

imgs_no = length(imgs_list);

% n_samples = size(samples_idx_struct.idxs,1);

p = samples_idx_struct.p;

ftrs_type = p.ftrs;
% 
% 
% switch imgs_list{1}(end - 2:end)
%     
%     case 'dcm'
%         
%         img = load_PA_data(imgs_list{1});
% 
%     case 'jpg'
%     
%         img = imread(imgs_list{1});
%         
%     case 'tif'
% 
%         img = imread(imgs_list{1});
%         
%         gt_img = imread(gts_list{1});
%         
%     case 'ppm'
%         
%         img = imread(imgs_list{1});
%         
%         gt_img = imread(gts_list{1});
%                 
%     otherwise
%         
% end
% 
% 
% 
% 
% switch gts_list{1}(end - 2:end)
%     
%     case 'dcm'
% 
%         gt_img = load_PA_data(gts_list{1});
% 
%     case 'jpg'
%     
%         gt_img = imread(gts_list{1});
%         
%         
%     case 'tif'
% 
%         gt_img = imread(gts_list{1});
%         
%     case 'ppm'
%         
%         gt_img = imread(gts_list{1});
% 
%         
%     case 'mat'
%         
%         gt_data = load(gts_list{1});
%         
%         gt_img = zeros(size(gt_data.seg));
%         
%         for il = 1 : length(gt_data.allshadow)
%             
%             gt_img(gt_data.seg == gt_data.allshadow(il)) = 1;
%             
%         end
%         
%         
%         for il = 1 : length(gt_data.allnonshadow)
%             
%             gt_img(gt_data.seg == gt_data.allnonshadow(il)) = -1;
%             
%         end
%         
%         
%     otherwise
%         
% end

trn_X = [];

trn_Y = [];

for i_img = 1 : imgs_no
    
    img_fn = imgs_list{i_img};
    
    switch img_fn(end - 2:end)
        
        case 'dcm'
            
            img = load_PA_data(img_fn);
            
        case 'jpg'
            
            img = imread(img_fn);
            
        case 'png'
            
            img = imread(img_fn);    
            
        case 'tif'
            
            img = imread(imgs_list{i_img});
            
        case 'ppm'
            
            img = imread(imgs_list{i_img});

        otherwise
            
    end
    
    gt_fn = gts_list{i_img};
    
    switch gt_fn(end - 2:end)
        
        case 'dcm'
            
            gt_img = load_PA_data(gt_fn);
            
        case 'jpg'
            
            gt_img = imread(gt_fn);
            
            
        case 'tif'
            
            gt_img = imread(gt_fn);
            
        case 'ppm'
            
            
        case 'png'    
            
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
    
    gt_img(gt_img > 0) = 1;
 
    idx_img = samples_idx_struct.idxs(samples_idx_struct.img_idxs == i_img,:);
    
    try
        
        pos_backslash = strfind(img_fn,'\');
        
        if(isempty(pos_backslash))
            
            pos_backslash = strfind(img_fn,'/');
            
        end
        
        img_nm = img_fn;
        
        img_nm(pos_backslash) = '_';
        
        img_nm = img_nm(pos_backslash(end - 1) + 1 : end);
        
        load(['../Cache/MICCAI_17_Shadow/' img_nm '_seg.mat']);
    
    catch
        
        disp 'meanshift segmentation'
    
        [dummy seg] = edison_wrapper(img, @RGB2Luv, ...
            'SpatialBandWidth', 9, 'RangeBandWidth', 15, ...
            'MinimumRegionArea', 200);
        
        seg = seg + 1;
        
        save(['../Cache/MICCAI_17_Shadow/' img_nm '_seg.mat'], 'seg');
        
    end
    
    numlabel = length(unique(seg(:)));
    
    
    try
        
        load(['../Cache/MICCAI_17_Shadow/'  img_nm '_labtexthist.mat']);
        

    catch exp1
        
        disp 'Single region feature'
        
        %load('cache/model_our.mat', 'model');
        labhist = calcLabHist(img, seg, numlabel);
      
        texthist = calcTextonHistNoInv(img, seg, numlabel);
        
        save([ '../Cache/MICCAI_17_Shadow/' img_nm '_labtexthist.mat'],...
            'labhist', 'texthist');
        
    end
    
    labels_samples = seg(idx_img);
    
    trn_ftrs = [labhist, texthist];
    
    trn_X_tmp = trn_ftrs(labels_samples,:);
    
    trn_Y_tmp = gt_img(idx_img);
    
    
    trn_X = [trn_X;trn_X_tmp];
    
    trn_Y = [trn_Y;trn_Y_tmp];
    
    
    
%     switch ftrs_type
%         
%         case 'RGB'
%             
%             for ib = 1 : size(img,3)
%                 
%                 tmp_img_cube = collect_patches(img(:,:,ib),idx_img,cube_size);
%                 
%                 n_ftrs_tmp = size(tmp_img_cube,2);
%                 
%                 img_cube(samples_idx_struct.img_idxs == i_img,n_ftrs : (n_ftrs + n_ftrs_tmp - 1)) = tmp_img_cube;
%                 
%                 n_ftrs = n_ftrs + n_ftrs_tmp;
%                 
%             end
%             
%         case 'Gabor'
%             
%             features = Gabor_Filter(img(:,:,2));
%        
%             img_cube(samples_idx_struct.img_idxs == i_img,:) = features(idx_img,:);
%             
%         
%        case 'Coye'     
%            
%            features = CoyeFilter(img);
%            
%            
%            img_cube(samples_idx_struct.img_idxs == i_img,:) = collect_patches(features,idx_img,[3,3]);
%            
%         otherwise
%             
%     end
%        
%     gt_cube(samples_idx_struct.img_idxs == i_img,:) = collect_patches(gt_img,idx_img,cube_size);
    
end
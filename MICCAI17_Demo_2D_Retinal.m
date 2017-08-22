%% This demo shows how to apply FOSP method on 2D retinal imaging

clear;

addpath(genpath('.\utils\'));

addpath(genpath('.\codes\'));

addpath(genpath('.\ext_codes\dollar_toolbox\'));

% in order to compare with two state-of-the-art semi sueprvised learning

% addpath(genpath('..\Ext_Codes\kde\'));

% addpath(genpath('..\Ext_Codes\svmlin\'));


data_type = 'DRIVE';

sub_data_type = [];

% now load the training and testing data

[trn_Y,trn_X,tst_Y,tst_X,dataInfo] = load_dataset(data_type,sub_data_type);


test_imgs_list = dataInfo.test_imgs_list;

test_gts_list = dataInfo.test_gts_list;

test_masks_list = dataInfo.test_masks_list;

p = dataInfo.p;


% train our semi-supervised method with only 500 labelled data

N_BaseLabels = 500;

% randomly pick 500 samples as the known labelled samples

label_idx = randi(length(trn_Y),N_BaseLabels,1);

% now train the baseline random forest 

pTrain = {'maxDepth',30,'M',100,'minChild',5,'H',2,'N1',N_BaseLabels};

forest_BL = forestTrain(trn_X(label_idx,:),trn_Y(label_idx),pTrain{:});





% now evaluate our method on individual testing data

for i_img = 1 : length(test_imgs_list)
    
    img = imread(test_imgs_list{i_img});
    
    gt_img = imread(test_gts_list{i_img});
    
    gt_img = gt_img > 0;
    
    mask_img = imread(test_masks_list{i_img});
    
    % remove the fringe area, a practice used by most of existing methods
    
    se = strel('disk',15);
    
    mask_img = imerode(mask_img > 0,se);
    
    
    imshow(img);
    
    title('Input Image');
    
    
  
    imshow(gt_img);
    
    title('Ground Truth Segmentation');
    
    
    
    [est_img_BL,tbc_img] = RF_predict_retinal_image(img,mask_img,...
        forest_BL,p);
    
    imshow(est_img_BL);
    
    title('Initial Estimation');
    
        
    
    % Now apply our method on the estimation score and corresponding
    % tree based code to generate the forest oriented superpixel(2D)
    
    [SP_map,cSP,Sp_info] = superpxiel_forest_oriented(est_img_BL,...
        double(tbc_img),2000,10,5);
   
    
    
    % After obtaining the super pixels, we collect 200 least confident super pixels 
    % as the candidates of suspicious regions. To save the efforts, we simply regard
    % all of the candidates as suspicious regions. We require the
    % suspicious region to be away from the boundary and the known
    % vessels( with extreme high vessel prediction)
    
    suspcious_region = collect_suspicious_region(est_img_BL,cSP,Sp_info,...
        SP_map,200);

    imshow(suspcious_region);
    
    title('Suspicious Super Pixels');
    
    
    
    % now train random forest to tell the suspicious region from the rest
    
    pTrain_sp = {'maxDepth',30,'M',100,'minChild',5,'H',2};
    
    forest_sp = RF_train_retinal_image(img,suspcious_region,mask_img,pTrain_sp,p);
    
    % now apply the learned classifier to predict the suspicious region.

    sp_prior = RF_predict_retinal_image(img,mask_img,...
        forest_sp,p);
    
    imshow(sp_prior);
    
    title('Low Confidence Region');    
    
        
    
    est_final = est_img_BL .* (1 - sp_prior);
    
    imshow(est_final);
    
    title('Final Estimation');    
 
end
 

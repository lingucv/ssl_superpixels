function p = setup_config_STARE()
% configuration parameters for the DRIVE vessel data 

%% dataset parameters

p.debug = 0;
% will run with debug parameter to check no errors will occur
%

p.dimension = 2; % dimension of images 

p.dataset_name = 'STARE'; % dataset name, used to find images lists

p.tr = 0;

p.color_img = 0;

p.patch_size = [15 15];

p.ftrs = 'RGB';

p.pos_sample_strategy = 'away_from_boundary+centreline';

p.pos_sample_distance = 1;

p.neg_sample_distance = 3;

p.results_dir = [];

% set up the train lists

% this part is specific for PA vessels

% base_folder = '..\Data/PA blood vessel model\thincup\';
% 
% indirs = dir([base_folder,'*UBP*']);

data_dir = '..\Data\STARE\stare_images\';

gt_dir = '..\Data\STARE\labels_ah\';

mask_dir = '..\Data\STARE\mask_imgs\';


files_list = dir([data_dir '*.ppm']);

img_list = 1:10;

for i_img = 1 : length(img_list)
    
    trn_img_list{i_img} = [data_dir  files_list(img_list(i_img)).name];
    
    trn_gt_list{i_img} = [gt_dir files_list(img_list(i_img)).name(1:6) '.ah.ppm'];
    
    trn_mask_list{i_img} = [mask_dir files_list(img_list(i_img)).name(1:6) '.png'];
    
end


p.trn_img_list = trn_img_list; 

p.trn_gt_list = trn_gt_list; 

p.trn_mask_list = trn_mask_list; 


% 
% data_dir = '..\Data\DRIVE\test\images\';
% 
% gt_dir = '..\Data\DRIVE\test\1st_manual\';
% 
% mask_dir = '..\Data\DRIVE\test\mask\';
% 
% 
% clear img_list;
% 
% img_list = {'01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' ...
%     '13' '14' '15' '16' '17' '18' '19' '20'};


img_list = 11:20;

for i_img = 1 : length(img_list)
    
    tst_img_list{i_img} = [data_dir  files_list(img_list(i_img)).name];
    
    tst_gt_list{i_img} = [gt_dir files_list(img_list(i_img)).name(1:6) '.ah.ppm'];
    
    tst_mask_list{i_img} = [mask_dir files_list(img_list(i_img)).name(1:6) '.png'];
    
end

p.tst_img_list = tst_img_list; 

p.tst_gt_list = tst_gt_list; 

p.tst_mask_list = tst_mask_list; 

%train lists 
%p.train_img_list_filename = fullfile('lists',p.dataset_name,'train_imgs.txt');
%p.train_radial_gt_list_filename = fullfile('lists',p.dataset_name,'train_gt_radial.txt');

%test lists
%p.test_img_list_filename = fullfile('lists',p.dataset_name,'test_imgs.txt');
%p.test_radial_gt_list_filename = fullfile('lists',p.dataset_name,'test_gt_radial.txt');


%% context features parameters
p.n_feat_center_img = uint32(50); % features on image computed at center
p.n_feat_cont_img = uint32(500); % features on image usign context
p.n_feat_center_ac = uint32(30); % features on score computed at center
p.n_feat_cont_ac = uint32(500); % features on score usign context
p.max_cont_step = int8(13); % max length of ray used for context features 

%precompute non-separable features for testing
p.n_precompute_features_image = 121;
p.n_precompute_features_ac = 121;

%% multiscale parameters
p.all_scales = single([5:10]); % total number of scales predicted (also used to compute gt) 
p.scale_toll = single(0.5*ones(size(p.all_scales))); % tolerance for discretizing scale domain

% 2 TRAIN SCALES
p.train_scales = single([6 9]); % scales used to train regressors
%p.predict_scales = {single([5:7]),single([8:10])}; % p.predict_scales{j} are scales predicted using regressor trained at scale p.train_scales(j) 
p.predict_scales = {single([6]),single([9])}; % p.predict_scales{j} are scales predicted using regressor trained at scale p.train_scales(j) 
p.predict_scales_test=p.predict_scales; %these should always be the same !
p.sample_scales= p.predict_scales; %


%% gradient boost parameters
%training samples
if(p.debug )
    p.neg_sample_no = 2000; %number of samples far from centerlines
    p.pos_sample_no = 2000; %number of samples close to centerlines
    p.T2Size = 1000; % number of samples used at each boosting iteration
else
    p.neg_sample_no = 20000; %number of samples far from centerlines
    p.pos_sample_no = 20000; %number of samples close to centerlines
    p.T2Size = 10000; % number of samples used at each boosting iteration
end

if(p.debug )
    p.iters_no = 25; % number of week learners ( i.e. number of boosting iterations )
    p.tree_depth_max = 1; % max deepth of a tree ( each tree is a weak learner )
    p.shrinkage_factor = 0.1; % shrinkage factor multiplied at score at each iteration
    p.loss_type = 'squared'; % loss to minimize at each iteration (on sampled pixels)

    %% autocontext parameters
    p.n_ac_iter = 1; % number of autoncontext iterations
else
    p.iters_no = 250; % number of week learners ( i.e. number of boosting iterations )
    p.tree_depth_max = 2; % max deepth of a tree ( each tree is a weak learner )
    p.shrinkage_factor = 0.1; % shrinkage factor multiplied at score at each iteration
    p.loss_type = 'squared'; % loss to minimize at each iteration (on sampled pixels)

    %% autocontext parameters
    p.n_ac_iter = 2; % number of autoncontext iterations
end
%% saving parameters
% p.codename = sprintf('%s_loss_%s_pos_%d_neg_%d_T2_%d_tree_depth_%d_boost_iters_%d_ac_iters_%d', ...
%     p.dataset_name,p.loss_type,p.pos_sample_no(1),p.neg_sample_no(1),p.T2Size, ...
%     p.tree_depth_max,p.iters_no,p.n_ac_iter);

%p.results_dir = fullfile('results',p.dataset_name,p.codename);

%% post processing parameters
p.do_nms = false; % do non-maxima suppression on images

%% multithread parameters
 p.omp_num_threads = '12';
 p.parfor_num_threads = 8;
 p.predict_parallel = 0;
 p.convolve_parallel = 0;
 

 %% multiscale last step parameters
 p.multiscale_ac  = 1;   
 
%train parameters
if(p.debug )
    p.n_iters_MC = 25;
    p.n_pos_MC = 10000;
    p.n_neg_MC = 10000;
else
    p.n_iters_MC = 100;
    p.n_pos_MC = 100000;
    p.n_neg_MC = 100000;
end
p.toll_pos_MC = 10;

%optimization params
p.opts_MC.loss = 'squaredloss'; 
p.opts_MC.shrinkageFactor = 0.1;
p.opts_MC.subsamplingFactor = 0.5;
p.opts_MC.maxTreeDepth= 2;
p.opts_MC.disableLineSearch = 0;
p.opts_MC.mtry = 300;

p.pooling_win_size_MC = [0 0 0 1 3 5 5 5 7 11]; % must be sorted
p.pooling_steps_MC = [0 1 2 0 4 0 4 8 0 4];

 

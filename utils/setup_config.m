function p = setup_config(data_type)

% configuration parameters 
%% dataset parameters

switch data_type
    
    case 'DRIVE'

        p.dimension = 2; % dimension of images
        
        p.dataset_name = 'DRIVE'; % dataset name, used to find images lists
        
        p.patch_size = [15 15];
        
        p.ftrs = 'RGB';
        
        p.pos_sample_strategy = 'away_from_boundary+centreline';
        
        p.pos_sample_distance = 1;
        
        p.neg_sample_distance = 3;
        
        p.results_dir = [];
        
        % set up the train lists
        
        data_dir = '.\data\DRIVE\training\images\';
        
        gt_dir = '.\data\DRIVE\training\1st_manual\';
        
        mask_dir = '.\data\DRIVE\training\mask\';
        
        
        img_list = 21:40;
        
        
        for i_img = 1 : length(img_list)
            
            trn_img_list{i_img} = [data_dir  num2str(img_list(i_img)) '_training.tif'];
            
            trn_gt_list{i_img} = [gt_dir num2str(img_list(i_img)) '_manual1.gif'];
            
            trn_mask_list{i_img} = [mask_dir num2str(img_list(i_img)) '_training_mask.gif'];
            
        end
        
        p.trn_img_list = trn_img_list;
        
        p.trn_gt_list = trn_gt_list;
        
        p.trn_mask_list = trn_mask_list;
        
        
        data_dir = '.\data\DRIVE\test\images\';
        
        gt_dir = '.\data\DRIVE\test\1st_manual\';
        
        mask_dir = '.\data\DRIVE\test\mask\';
        
        
        clear img_list;
        
        img_list = {'01' '02' '03' '04' '05' '06' '07' '08' '09' '10' '11' '12' ...
            '13' '14' '15' '16' '17' '18' '19' '20'};
        
        
        for i_img = 1 : length(img_list)
            
            tst_img_list{i_img} = [data_dir  img_list{i_img} '_test.tif'];
            
            tst_gt_list{i_img} = [gt_dir img_list{i_img} '_manual1.gif'];
            
            tst_mask_list{i_img} = [mask_dir img_list{i_img} '_test_mask.gif'];
            
        end
        
        p.tst_img_list = tst_img_list;
        
        p.tst_gt_list = tst_gt_list;
        
        p.tst_mask_list = tst_mask_list;

    otherwise
        
        
end

%training samples
p.neg_sample_no = 20000; %number of samples far from centerlines
p.pos_sample_no = 20000; %number of samples close to centerlines
 

function [trn_Y,trn_X,tst_Y,tst_X,dataInfo] = load_dataset(data_type,sub_name)
% load the dataset for the experiments of the MICCAI submission

switch data_type
    
    case 'DRIVE'
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config(data_type); % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_cache_directories(p); % create results directories
        
        %% load the data =====================================================
        
        fprintf('Loading the training and testing data\n');
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks(p,train_imgs_list,train_gts_list,train_masks_list);
        
        dataInfo.p = p;
                
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        % at first, extract the patch of RGB image as the raw features
        
        [trn_X_raw,trn_gt_samples] = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        % then collect Gabor feature, a kind of Wavelet features
        
        train_samples_idx_struct.p.ftrs = 'Gabor';
        
        trn_X_Gabor = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);

        % now collect Coye feature, a PCA based feature that is sensitive to the vessel
        
        train_samples_idx_struct.p.ftrs = 'Coye';
        
        trn_X_Coye = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        clear trn_gt_samples;
        
        
        trn_X = single([trn_X_raw,trn_X_Gabor,trn_X_Coye]);
        
                
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks(p,test_imgs_list,test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        
        [tst_X_raw,tst_gt_samples] = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        
        test_samples_idx_struct.p.ftrs = 'Gabor';
        
        tst_X_Gabor = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        test_samples_idx_struct.p.ftrs = 'Coye';
        
        tst_X_Coye = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);

        tst_X = single([tst_X_raw,tst_X_Gabor,tst_X_Coye]);
        
        
        
        tst_gt_samples = tst_gt_samples > 0;
        
        tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
        
        tst_Y = (tst_Y - 0.5) * 2;

        
    
    case 'STARE'    
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_STARE(); % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        % load the training data
        
        %% load the data =====================================================
        
        fprintf('Loading the training and testing data\n');
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks(p,train_imgs_list,train_gts_list,train_masks_list);
        
        dataInfo.p = p;
        
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        % 
        
        
        [trn_X_raw,trn_gt_samples] = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        
        train_samples_idx_struct.p.ftrs = 'Gabor';
        
        trn_X_Gabor = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        train_samples_idx_struct.p.ftrs = 'Coye';
        
        trn_X_Coye = load_samples_retinal(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        clear trn_gt_samples;
        
        
        trn_X = single([trn_X_raw,trn_X_Gabor,trn_X_Coye]);
        
        
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks(p,test_imgs_list,test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        
        
        [tst_X_raw,tst_gt_samples] = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        
        test_samples_idx_struct.p.ftrs = 'Gabor';
        
        tst_X_Gabor = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        test_samples_idx_struct.p.ftrs = 'Coye';
        
        tst_X_Coye = load_samples_retinal(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        
        
        tst_gt_samples = tst_gt_samples > 0;
        
        tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
        
        tst_Y = (tst_Y - 0.5) * 2;
        
        clear tst_gt_samples;
        
        tst_X = single([tst_X_raw,tst_X_Gabor,tst_X_Coye]);
        
        
     case 'Xray'
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_Xray_stereo_large_image_only(); % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        % load the training data
        
        %% load the data =====================================================
        
        fprintf('Loading the training and testing data\n');

        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks(p,train_imgs_list,train_gts_list,train_masks_list);
        
        dataInfo.p = p;
                
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        
        
        train_samples_idx_struct.p.ftrs = 'Grayscale';
        
        [trn_X_raw,trn_gt_samples] = load_samples_Xray(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        train_samples_idx_struct.p.ftrs = 'Gabor';
        
        trn_X_Gabor = load_samples_Xray(train_imgs_list,train_gts_list,train_samples_idx_struct);

        
        train_samples_idx_struct.p.ftrs = 'Coye';
        
        trn_X_Coye = load_samples_Xray(train_imgs_list,train_gts_list,train_samples_idx_struct);
        
        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        clear trn_gt_samples;
        
        
        trn_X = single([trn_X_raw,trn_X_Gabor,trn_X_Coye]);
        
                
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks(p,test_imgs_list,test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        
        test_samples_idx_struct.p.ftrs = 'Grayscale';
        
        [tst_X_raw,tst_gt_samples] = load_samples_Xray(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        
        test_samples_idx_struct.p.ftrs = 'Gabor';
        
        tst_X_Gabor = load_samples_Xray(test_imgs_list,test_gts_list,test_samples_idx_struct);
        
        test_samples_idx_struct.p.ftrs = 'Coye';
        
        tst_X_Coye = load_samples_Xray(test_imgs_list,test_gts_list,test_samples_idx_struct);

        
        
        tst_gt_samples = tst_gt_samples > 0;
        
        tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
        
        tst_Y = (tst_Y - 0.5) * 2;
        
        clear tst_gt_samples;
        
        tst_X = single([tst_X_raw,tst_X_Gabor,tst_X_Coye]);        
        
        
    case 'UIUC'
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_UIUC(); % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        % load the training data
        
        %% load the data =====================================================
        
        fprintf('Loading the training and testing data\n');
        
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks_v2(p,train_imgs_list,train_gts_list,train_masks_list);
        
        dataInfo.p = p;
                
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        [trn_X,trn_Y] = load_samples_shadow(train_imgs_list,...
            train_gts_list,train_samples_idx_struct);
        
        
        
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks_v2(p,test_imgs_list,test_gts_list,test_masks_list);
        
        dataInfo.p = p;
                
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        [tst_X,tst_Y] = load_samples_shadow(test_imgs_list,...
            test_gts_list,test_samples_idx_struct);        
 
        
%         trn_Y = trn_Y / 2 + 1.5;
%         
%         tst_Y = tst_Y / 2 + 1.5;
        
    case 'BigN'    
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_BigNeuron; % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks_BigNeuron(p,train_imgs_list,train_gts_list,train_masks_list);
        
        [trn_X,trn_gt_samples] = load_samples(train_imgs_list,...
            train_gts_list,train_samples_idx_struct);
        
        dataInfo.p = p;
        
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks_BigNeuron(p,test_imgs_list,...
            test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        [tst_X,tst_gt_samples] = load_samples(test_imgs_list,test_gts_list,...
            test_samples_idx_struct);

        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        
        
        tst_gt_samples = tst_gt_samples > 0;
        
        tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
        
        tst_Y = (tst_Y - 0.5) * 2;
        
        
        trn_X = single(trn_X);
        
        tst_X = single(tst_X);
        
        
    case 'BigN46'
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_BigNeuron_v3; % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks_BigNeuron(p,train_imgs_list,train_gts_list,train_masks_list);
        
        [trn_X,trn_gt_samples] = load_samples(train_imgs_list,...
            train_gts_list,train_samples_idx_struct);
        
        dataInfo.p = p;
        
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
        
        test_samples_idx_struct = get_samples_masks_BigNeuron(p,test_imgs_list,...
            test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
        dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
        [tst_X,tst_gt_samples] = load_samples(test_imgs_list,test_gts_list,...
            test_samples_idx_struct);
        
        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        
        
        tst_gt_samples = tst_gt_samples > 0;
        
        tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
        
        tst_Y = (tst_Y - 0.5) * 2;
        
        
        trn_X = single(trn_X);
        
        tst_X = single(tst_X);
        
    case 'BigN42'
        
        fprintf('Setting up configuration\n');
        
        [p] = setup_config_BigNeuron_v3; % load parameters (<--SET PARAMETERS HERE!)
        
        [p] = setup_directories_v2(p); % create results directories
        
        [train_imgs_list,train_gts_list,train_masks_list] = load_dataset_filelist(p,'train');
        
        train_samples_idx_struct = get_samples_masks_BigNeuron(p,train_imgs_list,train_gts_list,train_masks_list);
        
        [trn_X1,trn_gt_samples] = load_samples(train_imgs_list,...
            train_gts_list,train_samples_idx_struct);
        
        
%         train_samples_idx_struct.p.ftrs = 'Gabor';
%         
%         [trn_Gabor,trn_gt_samples1] = load_samples_3D(train_imgs_list,...
%             train_gts_list,train_samples_idx_struct);
        
        train_samples_idx_struct.p.ftrs = 'Coye';
        
        [trn_Coye,trn_gt_samples1] = load_samples_3D(train_imgs_list,...
            train_gts_list,train_samples_idx_struct);
        
        trn_X = [double(trn_X1),trn_Coye];
        
        dataInfo.p = p;
        
        dataInfo.train_imgs_list = train_imgs_list;
        
        dataInfo.train_gts_list = train_gts_list;
        
        dataInfo.train_masks_list = train_masks_list;
        
        dataInfo.train_samples_idx_struct = train_samples_idx_struct;
        
        
        [test_imgs_list,test_gts_list,test_masks_list] = load_dataset_filelist(p,'test');
%         
%         test_samples_idx_struct = get_samples_masks_BigNeuron(p,test_imgs_list,...
%             test_gts_list,test_masks_list);
        
        
        dataInfo.test_imgs_list = test_imgs_list;
        
        dataInfo.test_gts_list = test_gts_list;
        
        dataInfo.test_masks_list = test_masks_list;
        
%         dataInfo.test_samples_idx_struct = test_samples_idx_struct;
        
%         [tst_X,tst_gt_samples] = load_samples(test_imgs_list,test_gts_list,...
%             test_samples_idx_struct);
%         
        
        trn_gt_samples = trn_gt_samples > 0;
        
        trn_Y = trn_gt_samples(:,ceil(size(trn_gt_samples,2) / 2));
        
        trn_Y = (trn_Y - 0.5) * 2;
        
        trn_Y1 = trn_gt_samples(:,ceil(size(trn_gt_samples1,2) / 2));
        
        trn_Y1 = (trn_Y1 - 0.5) * 2;
        
        
        
        
        tst_X = [];
        
        tst_Y = [];
        
        trn_X = single(trn_X);
        
        
        
        
%         tst_gt_samples = tst_gt_samples > 0;
%         
%         tst_Y = tst_gt_samples(:,ceil(size(tst_gt_samples,2) / 2));
%         
%         tst_Y = (tst_Y - 0.5) * 2;
%         
%         
%         trn_X = single(trn_X);
%         
%         tst_X = single(tst_X);        
        
        
    case 'Adult'
        
        trn_data_fn = ['../Data/LibSVM/' sub_name];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
%         trn_X = trn_X * 255;
        
        tst_data_fn = ['../Data/LibSVM/' sub_name '.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        
%         tst_X = tst_X * 255;
        
    case 'Australian'
        
        trn_data_fn = ['../Data/LibSVM/australian'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        n_trn = length(trn_Y);
        
        trn_idx = randperm(length(trn_Y));

        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        
        tst_Y = trn_Y(1:floor(n_trn / 2),:);
       
        tst_X = trn_X(1:floor(n_trn / 2),:);
       
        trn_Y = trn_Y(floor(n_trn / 2) + 1 : end,:);
        
        trn_X = trn_X(floor(n_trn / 2) + 1 : end,:);
        
        
    case 'Breastcancer'
        
        trn_data_fn = ['../Data/LibSVM/breast-cancer'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        n_trn = length(trn_Y);
        
        trn_idx = randperm(length(trn_Y));
        
        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        trn_Y = trn_Y - 3;
        
        
        tst_Y = trn_Y(1:floor(n_trn / 2),:);
        
        tst_X = trn_X(1:floor(n_trn / 2),:);
        
        trn_Y = trn_Y(floor(n_trn / 2) + 1 : end,:);
        
        trn_X = trn_X(floor(n_trn / 2) + 1 : end,:);
        
        
    case 'Cod'
        
        trn_data_fn = ['../Data/LibSVM/cod-rna'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        
        tst_data_fn = ['../Data/LibSVM/cod-rna.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        

    case 'Covtype'
        
        trn_data_fn = ['../Data/LibSVM/covtype.libsvm.binary'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        trn_Y = (trn_Y - 1.5) * 2;
        
        n_trn = length(trn_Y);
        
        trn_idx = randperm(length(trn_Y));
        
        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        tst_Y = trn_Y(1:floor(n_trn / 2),:);
        
        tst_X = trn_X(1:floor(n_trn / 2),:);
        
        trn_Y = trn_Y(floor(n_trn / 2) + 1 : end,:);
        
        trn_X = trn_X(floor(n_trn / 2) + 1 : end,:); 
        
    case 'Epsilon'
        
        trn_data_fn = ['../Data/LibSVM/epsilon_normalized'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        trn_idx = randi(length(trn_Y),40000,1);
        
        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        
        tst_data_fn = ['../Data/LibSVM/epsilon_normalized.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        
        
        tst_idx = randi(length(tst_Y),40000,1);
        
        tst_X = tst_X(tst_idx,:);
        
        tst_Y = tst_Y(tst_idx,:);
        
        
    case 'Gisette'
        
        trn_data_fn = ['../Data/LibSVM/gisette_scale'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        tst_data_fn = ['../Data/LibSVM/gisette_scale.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        
        
    case 'HIGGS'
        
        trn_data_fn = ['../Data/LibSVM/HIGGS'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        n_trn = length(trn_Y);
        
        trn_Y = (trn_Y - 0.5) * 2;
        
%         trn_idx = randperm(length(trn_Y));
%         
%         trn_X = trn_X(trn_idx,:);
%         
%         trn_Y = trn_Y(trn_idx,:);
%         
        tst_Y = trn_Y(n_trn - 500000 + 1 : end,:);
        
        tst_X = trn_X(n_trn - 500000 + 1 : end,:);
        
        trn_Y = trn_Y(1 : n_trn - 500000,:);
        
        trn_X = trn_X(1 : n_trn - 500000,:);     
        
        
        trn_idx = randi(length(trn_Y),40000,1);
        
        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        
        tst_idx = randi(length(tst_Y),40000,1);
        
        tst_X = tst_X(tst_idx,:);
        
        tst_Y = tst_Y(tst_idx,:);
        
        
        
    case 'IJCNN1'
        
        trn_data_fn = ['../Data/LibSVM/ijcnn1.tr'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        tst_data_fn = ['../Data/LibSVM/ijcnn1.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        
        
    case 'Madelon'
        
        trn_data_fn = ['../Data/LibSVM/madelon'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        tst_data_fn = ['../Data/LibSVM/madelon.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);        
        
    case 'SUSY'
        
        trn_data_fn = ['../Data/LibSVM/SUSY'];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        n_trn = length(trn_Y);
        
        trn_Y = (trn_Y - 0.5) * 2;
        
%         trn_idx = randperm(length(trn_Y));
%         
%         trn_X = trn_X(trn_idx,:);
%         
%         trn_Y = trn_Y(trn_idx,:);
%         
        tst_Y = trn_Y(n_trn - 500000 + 1 : end,:);
        
        tst_X = trn_X(n_trn - 500000 + 1 : end,:);
        
        trn_Y = trn_Y(1 : n_trn - 500000,:);
        
        trn_X = trn_X(1 : n_trn - 500000,:);    
        
        
        trn_idx = randi(length(trn_Y),40000,1);
        
        trn_X = trn_X(trn_idx,:);
        
        trn_Y = trn_Y(trn_idx,:);
        
        
        tst_idx = randi(length(tst_Y),40000,1);
        
        tst_X = tst_X(tst_idx,:);
        
        tst_Y = tst_Y(tst_idx,:);
        
        
        
    case 'W'
        
        trn_data_fn = ['../Data/LibSVM/' sub_name];
        
        [trn_Y,trn_X] = libsvmread(trn_data_fn);
        
        
        tst_data_fn = ['../Data/LibSVM/' sub_name '.t'];
        
        [tst_Y,tst_X] = libsvmread(tst_data_fn);
        
    otherwise
        
        
end

n_ftr_trn = size(trn_X,2);

n_ftr_tst = size(tst_X,2);

% if(n_ftr_trn ~= n_ftr_tst)
% 
%    n_ftr = min([n_ftr_trn,n_ftr_tst]); 
%     
%    trn_X = trn_X(:,1:n_ftr);
%    
%    tst_X = tst_X(:,1:n_ftr);
%    
% end

% relevance_test(trn_X_Gabor,trn_Y,tst_X_Gabor,tst_Y,100,100);
% 
% relevance_test(trn_X_Coye,trn_Y,tst_X_Coye,tst_Y,100,100);

trn_Y = trn_Y * 0.5 + 1.5;

tst_Y = tst_Y * 0.5 + 1.5;

trn_X = full(trn_X);

tst_X = full(tst_X);


trn_Y = full(trn_Y);

tst_Y = full(tst_Y);


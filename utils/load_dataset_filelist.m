function [imgs_list,gts_list,masks_list] = load_dataset_filelist(p,operation)%,operation)
% load the file names of the imgs, gt and masks from txt lists containing path to images
%
%
if(strcmp(operation,'train'))
    imgs_list = p.trn_img_list;
    
    gts_list = p.trn_gt_list;
    
    if(isfield(p,'trn_mask_list'))
       
        masks_list = p.trn_mask_list;
        
    else
       
        masks_list = [];
        
    end
    
elseif(strcmp(operation,'test'))
    
    imgs_list =p.tst_img_list;
    
    gts_list = p.tst_gt_list;
    
    
    if(isfield(p,'tst_mask_list'))
       
        masks_list = p.tst_mask_list;
        
    else
       
        masks_list = [];
        
    end
    
  %  gts_list_fname = p.test_radial_gt_list_filename;
end

end

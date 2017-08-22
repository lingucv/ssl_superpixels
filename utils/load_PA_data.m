function X = load_PA_data(fn)

suffix_fn = fn(end - 2:end);

switch suffix_fn
    
    case 'dcm'
        
        X = dicomread(fn);
        
        X = permute(X,[1 2 4 3]);
        
    case 'tif'
        
        info = imfinfo(fn);
        
        num_imgs_tiff = numel(info);
        
        X_tmp = imread(fn,1);
        
        X = repmat(X_tmp,[1 1 num_imgs_tiff]);
        
        for i_b = 1 : num_imgs_tiff
            
            X(:,:,i_b) = imread(fn,i_b);
            
        end
        
        
    case 'iff'
        
        info = imfinfo(fn);
        
        num_imgs_tiff = numel(info);
        
        X_tmp = imread(fn,1);
        
        X = repmat(X_tmp,[1 1 num_imgs_tiff]);
        
        for i_b = 1 : num_imgs_tiff
            
            X(:,:,i_b) = imread(fn,i_b);
            
        end        
        
    otherwise
        
end
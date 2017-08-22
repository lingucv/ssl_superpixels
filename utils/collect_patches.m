function img_patch = collect_patches(img,idx,patch_size)

[h,w,b] = size(img);

% if(min(size(idx)) == 1)
%    
%     if()
%     
%     
%     is_vector = 1;
%     
%     
%     
% else
%     
%     is_vector = 0;
%     
% end

% regulate the input index

n_sample = length(idx);

if((n_sample > 3) && (size(idx,1) ~= n_sample))
   
    idx = idx';
    
end    
    


if(b > 1)
    
    % at exam whether idx is N * 1 or N * dim
    
    if(size(idx,2) == 3)
        
        x = idx(:,1);
        
        y = idx(:,2);
        
        z = idx(:,3);
        
    else
        
        [x,y,z] = ind2sub(size(img),idx);
        
    end
    
    patch_radius = floor(patch_size / 2);
    
    
    
    [x_offset,y_offset,z_offset] = meshgrid(-patch_radius(1) : patch_radius(1), -patch_radius(2) : patch_radius(2),...
        -patch_radius(3)  : patch_radius(3));
    
    x_offset = x_offset(:)';
    
    y_offset = y_offset(:)';
    
    z_offset = z_offset(:)';
    
    
    
    x_m = repmat(x,[1, length(x_offset)]) + repmat(x_offset,[n_sample,1]);
    
    y_m = repmat(y,[1, length(y_offset)]) + repmat(y_offset,[n_sample,1]);
    
    z_m = repmat(z,[1, length(z_offset)]) + repmat(z_offset,[n_sample,1]);
    
    [m,n,k] = size(img);
    
    x_m = min(x_m,m);
    
    x_m = max(x_m,1);
    
    
    
    y_m = min(y_m,n);
    
    y_m = max(y_m,1);    
    
    
    
    z_m = min(z_m,k);
    
    z_m = max(z_m,1);    
    
    
    ind_m = sub2ind(size(img),x_m,y_m,z_m);
    
    img_patch = img(ind_m);
    
else
    
    if(size(idx,2) == 2)
        
        x = idx(:,1);
        
        y = idx(:,2);
        
    else
        
        [x,y] = ind2sub(size(img),idx);
        
    end
    
    patch_radius = floor(patch_size / 2);
    
    [x_offset,y_offset] = meshgrid(-patch_radius(1) : patch_radius(1), -patch_radius(2) : patch_radius(2));
    
    x_offset = x_offset(:)';
    
    y_offset = y_offset(:)';
    
    
    x_m = repmat(x,[1, prod(patch_size)]) + repmat(x_offset,[n_sample,1]);
    
    y_m = repmat(y,[1, prod(patch_size)]) + repmat(y_offset,[n_sample,1]);
    
    
    [m,n,k] = size(img);
    
    x_m = min(x_m,m);
    
    x_m = max(x_m,1);
    
    
    
    y_m = min(y_m,n);
    
    y_m = max(y_m,1);
    
    
    ind_m = sub2ind(size(img),x_m,y_m);
    
    img_patch = img(ind_m);
    
end






    
% 
% 
% switch b
%    
%     case 3
%     
%         if()
%         
%     otherwise
%         
% end
%         
        

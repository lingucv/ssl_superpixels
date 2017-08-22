function [data,idx] = data_sample(data,k,dim)

if(min(size(data)) == 1)

    idx = randi(length(data),k,1);
    
    data = data(idx);

else
    
    if(nargin < 3)
        
       dim = 1;
        
    end
    
    idx = randi(size(data,dim),k,1);
    
    if(dim == 1)
        
        data = data(idx,:);
        
    end
    
    if(dim == 2)
        
        data = data(:,idx);
        
    end
    
end

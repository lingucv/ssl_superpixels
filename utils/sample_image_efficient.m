function sample_idx = sample_image_efficient(roi_mask,N_samples)

% provide a fast and efficient method to randomly sample when the number of
% the candidates are too huge

roi_mask = roi_mask > 0;

if(mean(roi_mask(:)) > 0.3)
    
    idx_range = numel(roi_mask);
    
    candidate_idx = randi(idx_range,N_samples * 3,1);
    
    chosen_candidate = roi_mask(candidate_idx);
    
    candidate_idx = candidate_idx(chosen_candidate);
    
else
    
    candidate_idx = find(roi_mask(:));
        
end

sample_idx = data_sample(candidate_idx,N_samples);

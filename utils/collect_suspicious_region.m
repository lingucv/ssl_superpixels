function suspicious_region = collect_suspicious_region(est_img,cSP,Sp_info,...
        SP_map,nCandidates)

thres_est = 0.75;
    
[~,idx_SP_cand] = sort(cSP);

nCandidates = min(nCandidates, length(cSP));

idx_SP_cand = idx_SP_cand(1:nCandidates);

susp_SP_cand = (Sp_info.dist_kf_SP(idx_SP_cand) > 3) & (Sp_info.dist_mask_SP(idx_SP_cand) > 10);

sp_idx = idx_SP_cand(susp_SP_cand);    


suspicious_region = zeros(size(est_img));

for iSP = 1 : length(sp_idx)
    
    new_SP_idxs = find(SP_map == sp_idx(iSP));
    
    est_SP = est_img(new_SP_idxs);
    
    new_SP_idxs(est_SP > thres_est) = [];
    
    suspicious_region(new_SP_idxs) = 1;
    
end
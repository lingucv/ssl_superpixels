function forest = SSL_forestTrain_KNN_v3( data, hs, udata, uhs, varargin )
% Train a semi supervised random forest classifier.
% The current algorithm approximates the verification by the KNN method at
% first

% now add the debug mode to do the real time evaluation

%
% Dimensions:
%  M - number trees
%  F - number features
%  N - number input vectors
%  H - number classes
%
% USAGE
%  forest = forestTrain( data, hs, [varargin] )
%
% INPUTS
%  data     - [NxF] N length F feature vectors
%  hs       - [Nx1] or {Nx1} target output labels in [1,H]
%  varargin - additional params (struct or name/value pairs)
%   .M          - [1] number of trees to train
%   .H          - [max(hs)] number of classes
%   .N1         - [5*N/M] number of data points for training each tree
%   .F1         - [sqrt(F)] number features to sample for each node split
%   .split      - ['gini'] options include 'gini', 'entropy' and 'twoing'
%   .minCount   - [1] minimum number of data points to allow split
%   .minChild   - [1] minimum number of data points allowed at child nodes
%   .maxDepth   - [64] maximum depth of tree
%   .dWts       - [] weights used for sampling and weighing each data point
%   .fWts       - [] weights used for sampling features
%   .discretize - [] optional function mapping structured to class labels
%                    format: [hsClass,hBest] = discretize(hsStructured,H);
%
% OUTPUTS
%  forest   - learned forest model struct array w the following fields
%   .fids     - [Kx1] feature ids for each node
%   .thrs     - [Kx1] threshold corresponding to each fid
%   .child    - [Kx1] index of child for each node
%   .distr    - [KxH] prob distribution at each node
%   .hs       - [Kx1] or {Kx1} most likely label at each node
%   .count    - [Kx1] number of data points at each node
%   .depth    - [Kx1] depth of each node
%
% EXAMPLE
%  N=10000; H=5; d=2; [xs0,hs0,xs1,hs1]=demoGenData(N,N,H,d,1,1);
%  xs0=single(xs0); xs1=single(xs1);
%  pTrain={'maxDepth',50,'F1',2,'M',150,'minChild',5};
%  tic, forest=forestTrain(xs0,hs0,pTrain{:}); toc
%  hsPr0 = forestApply(xs0,forest);
%  hsPr1 = forestApply(xs1,forest);
%  e0=mean(hsPr0~=hs0); e1=mean(hsPr1~=hs1);
%  fprintf('errors trn=%f tst=%f\n',e0,e1); figure(1);
%  subplot(2,2,1); visualizeData(xs0,2,hs0);
%  subplot(2,2,2); visualizeData(xs0,2,hsPr0);
%  subplot(2,2,3); visualizeData(xs1,2,hs1);
%  subplot(2,2,4); visualizeData(xs1,2,hsPr1);
%
% See also forestApply, fernsClfTrain
%
% Piotr's Computer Vision Matlab Toolbox      Version 3.24
% Copyright 2014 Piotr Dollar.  [pdollar-at-gmail.com]
% Licensed under the Simplified BSD License [see external/bsd.txt]

% get additional parameters and fill in remaining parameters
dfs={ 'M',1, 'H',[], 'N1',[], 'F1',[], 'split','gini', 'minCount',1, ...
  'minChild',1, 'maxDepth',64, 'dWts',[], 'fWts',[], 'discretize','','SSL',0,'debug',0,...
  'record',0,'optimal',0,'expSSL',0};
[M,H,N1,F1,splitStr,minCount,minChild,maxDepth,dWts,fWts,discretize,SSL,...
    debug_mode,record,optimal,expSSL] = ...
  getPrmDflt(varargin,dfs,1);
[N,F]=size(data); assert(length(hs)==N); discr=~isempty(discretize);
minChild=max(1,minChild); minCount=max([1 minCount minChild]);
if(isempty(H)), H=max(hs); end; assert(discr || all(hs>0 & hs<=H));
if(isempty(N1)), N1=round(5*N/M); end; N1=min(N,N1);
if(isempty(F1)), F1=round(sqrt(F)); end; F1=min(F,F1);
if(isempty(dWts)), dWts=ones(1,N,'single'); end; dWts=dWts/sum(dWts);
if(isempty(fWts)), fWts=ones(1,F,'single'); end; fWts=fWts/sum(fWts);
split=find(strcmpi(splitStr,{'gini','entropy','twoing'}))-1;
if(isempty(split)), error('unknown splitting criteria: %s',splitStr); end

% make sure data has correct types
if(~isa(data,'single')), data=single(data); end
if(~isa(hs,'uint32') && ~discr), hs=uint32(hs); end
if(~isa(udata,'single')), udata = single(udata); end
if(~isa(uhs,'uint32') && ~discr), uhs=uint32(uhs); end
if(~isa(fWts,'single')), fWts=single(fWts); end
if(~isa(dWts,'single')), dWts=single(dWts); end

% train M random trees on different subsets of data
prmTree = {H,F1,minCount,minChild,maxDepth,fWts,split,discretize,SSL,...
    debug_mode,record,optimal,expSSL};

for i=1:M
  if(N==N1), data1=data; hs1=hs; dWts1=dWts; udata1 = udata; uhs1 = uhs; else
    d=wswor(dWts,N1,4); data1=data(d,:); hs1=hs(d);
    dWts1=dWts(d); dWts1=dWts1/sum(dWts1); 
    
    N_udata = size(udata,1);
    
    N1_udata = round(5 * N_udata / M);
    
    N1_udata = min(N1_udata,size(udata,1));
    
    du = randi(N_udata,N1_udata,1);
    
    udata1 = udata(du,:);
    
    uhs1 = uhs(du);
    
    %udata1 = udata(d,:)
  end
  tree = treeTrain(data1,hs1,udata1,uhs1,prmTree);
  if(i==1), forest=tree(ones(M,1)); else forest(i)=tree; end
  
  %forest(i) = tree;
  
end

end

function tree = treeTrain( data, hs, udata,uhs, prmTree )
% Train single random tree.
[H,F1,minCount,minChild,maxDepth,fWts,split,discretize,SSL,debug_mode,...
    record,optimal,expSSL]=deal(prmTree{:});



if(SSL)
    
    tree_mode = 2;
    
else
    
    tree_mode = 1;
    
    
end

if(optimal == 1)
    
    tree_mode = 9;
    
end


if(optimal == 2)
    
    tree_mode = 8;
    
end

if(expSSL)
    
    tree_mode = 10 + expSSL;
    
end


if(tree_mode == 14)
    
    F2 = 100;
    
    fid_knn = wswor(fWts,F2,4);
    
    [hs_KNN,hs_KNN_dis] = KNN_label2unlabel(data(:,fid_knn),hs,udata(:,fid_knn));
    
    hs_KNN_dis_sort = sort(hs_KNN_dis);
    
    N=size(data,1);
    
    l_knn = max(ceil(0.05 * length(hs_KNN_dis_sort)), 1.5 * N);
    
%     hs_KNN_dis
%     
    
%     l_knn = ceil(0.05 * length(hs_KNN_dis_sort));
%     
     thr_KNN = hs_KNN_dis_sort(l_knn);
     
      KNN_mask = (hs_KNN_dis' > thr_KNN);
      
      KNN_mask = ~KNN_mask;
      
%     
%     if(thr_KNN < 1)
%     
%         
%         
% 
%     end
        
    err_KNN1 = mean(hs_KNN ~= uhs);
%     
%     KNN_mask = (hs_KNN_dis' < thr_KNN);
%     
    err_KNN2 = mean(hs_KNN(KNN_mask) ~= uhs(KNN_mask));
    
    [err_KNN1, err_KNN2]

    
    udata = udata(KNN_mask,:);
    
    uhs = uhs(KNN_mask);
    
    hsk = hs_KNN(KNN_mask);
    
end

N=size(data,1); K=2*size(udata,1)-1; discr=~isempty(discretize);
thrs=zeros(K,1,'single'); distr=zeros(K,H,'single');
fids=zeros(K,1,'uint32'); child=fids; count=fids; depth=fids;

countu = count;

hsn=cell(K,1); dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;

Nu = size(udata,1);

didus = cell(K,1);

didus{1} = uint32(1:Nu);


while( k < K )
    % get node data and store distribution
    dids1=dids{k}; dids{k}=[]; hs1=hs(dids1); n1=length(hs1); count(k)=n1;
    
    didus1 = didus{k}; didus{k}=[]; 
    
    hsk1 = hsk(didus1);  nu1 = length(hsk1); countu(k) = nu1;
    
    if(~isempty(hs1))
    
        purel = all(hs1(1)==hs1);
    
    else
        
        purel = 1;
        
    end
        
        
    purek = all(hsk1(1) == hsk1);
    
    if(purel)
        
        switch tree_mode
            
            case 14
                
                if(purek || (isempty(hs1)))
                    
                    distr(k,:)=histc(hsk1,1:H) / nu1;
                    
                    [~,hsn{k}]=max(distr(k,:));
                    
                end
                
            otherwise
                
                distr(k,hs1(1))=1; 
        
                hsn{k}=hs1(1);
                
        end
        
    else
        
        distr(k,:)=histc(hs1,1:H)/n1;
        
        [~,hsn{k}]=max(distr(k,:));
        
    end
    
    % if pure node or insufficient data don't train split
    
    switch tree_mode
        
        case 14
            
            if( purek || nu1<=minCount || depth(k)>maxDepth ), k=k+1; continue; end
            
        otherwise
           
            if( purel || n1<=minCount || depth(k)>maxDepth ), k=k+1; continue; end
    
    end
    
    
    % train split and continue
    fids1=wswor(fWts,F1,4); data1=data(dids1,fids1);
    %[~,order1]=sort(data1); order1=uint32(order1-1);
    
    datau1 = udata(didus1,fids1);
    
    uhs1 = uhs(didus1);
    
    
    switch tree_mode
        
        case 14
            
            if(purel || n1 < minCount)
            
                 [fid,thr,gain] = SSLforestFindThr(datau1,hsk1,datau1,uhs1,tree_mode);
            
            else
                
                 [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,tree_mode);
               
                
            end
                
        otherwise
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,tree_mode);
            
            
    end
    
    if(record)
        
        datal{k} = data1;
        
        hsl{k} = hs1;
        
        datau{k} = datau1;
        
        hsu{k} = uhs1;
        
    end
    
    fid=fids1(fid); 
%     
%     switch tree_mode
%         
%         case 14
%             
%             if(purel || n1 < minCount)
%             
%             else
%                 
%                left = data(dids1,fid)<thr;
%                
%                count0 = nnz(left);
%                 
%             end
%  
%             
%         otherwise
%             
%                left=data(dids1,fid)<thr; 
%                
%                count0=nnz(left);
%             
%     end

    
    left = data(dids1,fid)<thr; count0=nnz(left);
    
    leftu = udata(didus1,fid) < thr; countu0 = nnz(leftu);
    
    switch tree_mode
        
        case 14
            
            if( gain>1e-10 && countu0>=minChild && (nu1-countu0)>=minChild )
                child(k)=K; fids(k)=fid-1; thrs(k)=thr;
                dids{K}=dids1(left); dids{K+1}=dids1(~left);
                
                didus{K} = didus1(leftu); didus{K + 1} = didus1(~leftu);
                
                depth(K:K+1)=depth(k)+1; K=K+2;
            end
            
        otherwise
            
            if( gain>1e-10 && count0>=minChild && (n1-count0)>=minChild )
                child(k)=K; fids(k)=fid-1; thrs(k)=thr;
                dids{K}=dids1(left); dids{K+1}=dids1(~left);
                
                didus{K} = didus1(leftu); didus{K + 1} = didus1(~leftu);
                
                depth(K:K+1)=depth(k)+1; K=K+2;
            end
            
    end
    
    k=k+1;
end
% create output model struct
K=1:K-1;

hsn=[hsn{K}]';
 
 
tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
    'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K));


if(record)
    
   tree.datau = datau;
   
   tree.hsu = hsu;
   
   tree.datal = datal;
   
   tree.hsl = hsl;
    
end

end




function ids = wswor( prob, N, trials )
% Fast weighted sample without replacement. Alternative to:
%  ids=datasample(1:length(prob),N,'weights',prob,'replace',false);
M=length(prob); assert(N<=M); if(N==M), ids=1:N; return; end
if(all(prob(1)==prob)), ids=randperm(M,N); return; end
cumprob=min([0 cumsum(prob)],1); assert(abs(cumprob(end)-1)<.01);
cumprob(end)=1; [~,ids]=histc(rand(N*trials,1),cumprob);
[s,ord]=sort(ids); K(ord)=[1; diff(s)]~=0; ids=ids(K);
if(length(ids)<N), ids=wswor(cumprob,N,trials*2); end
ids=ids(1:N)';
end

function [fid,thr,gain] = SSLforestFindThr(data1,hs1,udata1,uhs1,tree_mode)

if(nargin < 5)
    
   tree_mode = 1; 
    
end



Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);

g_gain = zeros(size(data1,2),1);

thr_c = zeros(size(data1,2),1);

N = size(udata1,1);

[data1_order,order1] = sort(data1);

[gini,gini_initial] = Calculate_Gini_Distr_Fast(data1,hs1,order1);

nftrs = size(data1,2);

switch tree_mode
    
    case 1
        
        [gain,min_idx] = min(gini(:));
        
        [fid,thr_l_idx] = ind2sub(size(gini),min_idx); 
    
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr = data1_order(thr_l_idx,fid); 
        
        gain = gini_initial - gain;
        
    case 2
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        if(v_gini > 0.15)
           
            hs_KNN = KNN_label2unlabel(data1,hs1,udata1);
            
            gain_KNN = evaluate_argument_data(udata1,hs_KNN,1 : nftrs,thr_l);
            
            [~,fid] = max(gain_KNN);
            
            thr = thr_l(fid);
           
            gain = gain_l(fid);
            
            
        else
            
            [gain,fid] = max(gain_l);
            
            thr = thr_l(fid);
            
            
            
        end
        
    case 8
        
        [data1_order,order1] = sort(udata1);
        
        [gini,gini_initial] = Calculate_Gini_Distr_Fast(udata1,uhs1,order1);
        
        [gain,min_idx] = min(gini(:));
        
        [fid,thr_l_idx] = ind2sub(size(gini),min_idx);
        
        thr_l_idx = min(thr_l_idx + 1,size(udata1,1));
        
        thr = data1_order(thr_l_idx,fid);
        
        gain = gini_initial - gain;
        
        
    case 9
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        gain_optimal = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
        
        [~,fid] = max(gain_optimal);
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        
        
    case 11
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        
        
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        gain_l = gini_initial - gain_l;
        
        
        [~,fid_c] = sort(gain_l,'descend');
        
        fid_c = fid_c(1:20);
        
        gain_optimal = evaluate_argument_data(udata1(:,fid_c),uhs1,1 : 20,thr_l(fid_c));
        
        [~,fid] = max(gain_optimal);
        
        fid = fid_c(fid);
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
    case 14
        
        [gain,min_idx] = min(gini(:));
        
        [fid,thr_l_idx] = ind2sub(size(gini),min_idx);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr = data1_order(thr_l_idx,fid);
        
        gain = gini_initial - gain;
        
        
    otherwise
        
end

end



function [fid,thr,gain] = SSLforestFindThr_KNN(data1,hs1,udata1,uhs1,hs1_KNN,KNN_mask,tree_mode)

if(nargin < 6)
    
   tree_mode = 1; 
    
end

Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);

g_gain = zeros(size(data1,2),1);

thr_c = zeros(size(data1,2),1);

N = size(udata1,1);

[data1_order,order1] = sort(data1);

[gini,gini_initial] = Calculate_Gini_Distr_Fast(data1,hs1,order1);

nftrs = size(data1,2);

switch tree_mode
    
    case 1
        
        [gain,min_idx] = min(gini(:));
        
        [fid,thr_l_idx] = ind2sub(size(gini),min_idx); 
    
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr = data1_order(thr_l_idx,fid); 
        
        gain = gini_initial - gain;
        
    case 2
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        if(v_gini > 0.2)
           
            hs_KNN = KNN_label2unlabel(data2,hs1,udata2);
            
            gain_KNN = evaluate_argument_data(udata1,hs_KNN,1 : nftrs,thr_l);
            
            [~,fid] = max(gain_KNN);
            
            thr = thr_l(fid);
           
            gain = gain_l(fid);
            
            
        else
            
            [gain,fid] = max(gain_l);
            
            thr = thr_l(fid);
            
            
            
        end
        
    case 8
        
        [data1_order,order1] = sort(udata1);
        
        [gini,gini_initial] = Calculate_Gini_Distr_Fast(udata1,uhs1,order1);
        
        [gain,min_idx] = min(gini(:));
        
        [fid,thr_l_idx] = ind2sub(size(gini),min_idx);
        
        thr_l_idx = min(thr_l_idx + 1,size(udata1,1));
        
        thr = data1_order(thr_l_idx,fid);
        
        gain = gini_initial - gain;
        
        
    case 9
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        gain_optimal = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
        
        [~,fid] = max(gain_optimal);
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        
        
    case 11
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        
        
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        gain_l = gini_initial - gain_l;
        
        
        [~,fid_c] = sort(gain_l,'descend');
        
        fid_c = fid_c(1:20);
        
        gain_optimal = evaluate_argument_data(udata1(:,fid_c),uhs1,1 : 20,thr_l(fid_c));
        
        [~,fid] = max(gain_optimal);
        
        fid = fid_c(fid);
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        
    case 12
        
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        if(v_gini > 0)
           
%             hs_KNN = KNN_label2unlabel(data2,hs1,udata2);
            
            gain_u = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);

            gain_KNN = evaluate_argument_data(udata1,hs1_KNN,1 : nftrs,thr_l);

            
            gain_l_lb_mask = gain_l > (max(gain_l) - (v_gini / 2)); 
            
            gain_l_lb = gain_l;
            
            gain_l_lb(gain_l_lb_mask) = gain_l_lb(gain_l_lb_mask) - (v_gini / 2);
            
            gain_l_lb(gain_l_lb < min(gain_l)) = min(gain_l);
            
            
            plot(1:nftrs,gain_KNN,'r',1:nftrs,gain_u,'g',1:nftrs,gain_l,'b',1:nftrs,gain_u + v_gini,'b--');
            
            
%             gain_u = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
            
            [Nl1 Nl2 sum(uhs1 == 1) sum(uhs1 == 2)]
            
            [mean(hs1_KNN(uhs1 == 1) ~= 1),mean(hs1_KNN(uhs1 == 2) ~= 2)]
            
            [~,fid_lo] = max(gain_l); 
            
            gainlo = gain_u(fid_lo);
            
            [~,fid_KNN] = max(gain_KNN);
            
            gainKNN = gain_u(fid_KNN);
            
            
            [gainlo,gainKNN,max(gain_u)]
            
            Nk1 = sum(hs1_KNN == 1) / length(hs1_KNN) * length(hs1);
            
            Nk2 = sum(hs1_KNN == 2) / length(hs1_KNN) * length(hs1);
            
            [Nl1 Nl2 Nk1 Nk2]
            
            
            
%             [~,fid_c] = sort(gain_u,'descend');
%             
%             gain_l(fid_c(1:3))
%             
%             gain_u(fid_c(1:3))
            
            v_gini
            
            [~,fid] = max(gain_KNN);
            
            thr = thr_l(fid);
           
            gain = gain_l(fid);
            
            
        else
            
            [gain,fid] = max(gain_l);
            
            thr = thr_l(fid);
            
            
            
        end
        
        
    otherwise
        
end

end

function gain = calculate_gini_gain(data,hs,fid,thr)

left = data(:,fid)<thr;

gain = get_gini_gain(hs,left);

end

function gini_v = approximate_gini_v(hs1)

Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);

Nl = Nl1 + Nl2;

p1 = Nl1 / (Nl + eps);

p2 = Nl2 / (Nl + eps);

gini_v = sqrt(p1 .* p2 / Nl + 1 ./ (max(Nl,1) ^ 2 ));


end

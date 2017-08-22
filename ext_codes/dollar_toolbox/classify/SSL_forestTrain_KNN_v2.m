function forest = SSL_forestTrain_KNN_v2( data, hs, udata, uhs, varargin )
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
end

end

function tree = treeTrain( data, hs, udata,uhs, prmTree )
% Train single random tree.
[H,F1,minCount,minChild,maxDepth,fWts,split,discretize,SSL,debug_mode,...
    record,optimal,expSSL]=deal(prmTree{:});
N=size(data,1); K=2*N-1; discr=~isempty(discretize);
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
    
    didus1 = didus{k}; didus{k}=[]; nu1 = size(didus1,1); countu(k) = nu1;
    
    
    if(discr), [hs1,hsn{k}]=feval(discretize,hs1,H); hs1=uint32(hs1); end
    if(discr), assert(all(hs1>0 & hs1<=H)); end; pure=all(hs1(1)==hs1);
    if(~discr), if(pure), distr(k,hs1(1))=1; hsn{k}=hs1(1); else
            distr(k,:)=histc(hs1,1:H)/n1; [~,hsn{k}]=max(distr(k,:)); end; end
    % if pure node or insufficient data don't train split
    if( pure || n1<=minCount || depth(k)>maxDepth ), k=k+1; continue; end
    % train split and continue
    fids1=wswor(fWts,F1,4); data1=data(dids1,fids1);
    [~,order1]=sort(data1); order1=uint32(order1-1);
    
    datau1 = udata(didus1,fids1);
    
    uhs1 = uhs(didus1);
    
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
    
    switch tree_mode
            
        case 13
            
            F2 = 100;
            
            fid_knn = wswor(fWts,F2,4);
            
            [hs_KNN,~] = KNN_label2unlabel(data(dids1,fid_knn),hs1,udata(didus1,fid_knn));
            
            [fid,thr,gain] = SSLforestFindThr_KNN2(data1,hs1,datau1...
                ,uhs1,hs_KNN,true(size(uhs1)),tree_mode);
            
            
        otherwise
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,tree_mode,debug_mode);
            
    end
        
    if(record)
    
        datal{k} = data1;
        
        hsl{k} = hs1;
        
        datau{k} = datau1;
        
        hsu{k} = uhs1;
        
    end

  fid=fids1(fid); left=data(dids1,fid)<thr; count0=nnz(left);
  
  leftu = udata(didus1,fid) < thr; 
  
  if( gain>1e-10 && count0>=minChild && (n1-count0)>=minChild )
    child(k)=K; fids(k)=fid-1; thrs(k)=thr;
    dids{K}=dids1(left); dids{K+1}=dids1(~left);
    
    didus{K} = didus1(leftu); didus{K + 1} = didus1(~leftu);
    
    depth(K:K+1)=depth(k)+1; K=K+2;
  end; k=k+1;
end
% create output model struct
K=1:K-1; if(discr), hsn={hsn(K)}; else hsn=[hsn{K}]'; end

if(debug_mode)
    
    tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
        'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K),'gain_l',gain_l,...
        'gain_w1',gain_w1,'gain_w2',gain_w2);
    
else
    
    tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
        'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K));
    
end


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




function [fid,thr,gain] = SSLforestFindThr_KNN2(data1,hs1,udata1,uhs1,hs1_KNN,KNN_mask,tree_mode)

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
    
        
    case 13
        
        Nu1 = sum(uhs1 == 1);
        
        Nu2 = sum(uhs1 == 2);
        
        Nu = Nu1 + Nu2;
        
        Nl = Nl1 + Nl2;
        
        
        
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        [gain_u,bl_l1,bl_l2,bl_r1,bl_r2] = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
        
        [gain_l1,l_l1,l_l2,l_r1,l_r2] = evaluate_argument_data(data1,hs1,1 : nftrs,thr_l);
        
%         gain_KNN = evaluate_argument_data(udata1,hs1_KNN,1 : nftrs,thr_l);
        
        
        plot(1:nftrs,gain_l,'r',1:nftrs,gain_u,'g');
        
        [gain,fid] = max(gain_l);
        
       [~,fid_u] = max(gain_u);
        
        [gain_u(fid) max(gain_u)]
        
        [([bl_l1(fid),bl_l2(fid),bl_r1(fid),bl_r2(fid)]) / Nu * Nl;...
            [l_l1(fid),l_l2(fid),l_r1(fid),l_r2(fid)]]
        
        [([bl_l1(fid_u),bl_l2(fid_u),bl_r1(fid_u),bl_r2(fid_u)]) / Nu * Nl;...  
        
        [l_l1(fid_u),l_l2(fid_u),l_r1(fid_u),l_r2(fid_u)]]
        
        
        thr = thr_l(fid);
        
        
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

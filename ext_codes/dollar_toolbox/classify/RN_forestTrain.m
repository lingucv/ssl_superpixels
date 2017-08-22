function forest = RN_forestTrain( data, hs, udata, uhs, varargin )
% Train a semi supervised random forest classifier.
% This function implements the robust node splitting algorithm reported in
% TIP 

% make some revisement to achieve the comparable performance as the
% baseline

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
  'minChild',1, 'maxDepth',64, 'dWts',[], 'fWts',[], 'discretize',''};
[M,H,N1,F1,splitStr,minCount,minChild,maxDepth,dWts,fWts,discretize] = ...
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
prmTree = {H,F1,minCount,minChild,maxDepth,fWts,split,discretize};

for i=1:M
  if(N==N1), data1=data; hs1=hs; dWts1=dWts; udata1 = udata; uhs1 = uhs; else
    d=wswor(dWts,N1,4); data1=data(d,:); hs1=hs(d);
    dWts1=dWts(d); dWts1=dWts1/sum(dWts1); 
    
    N_udata = size(udata,1);
    
    if(N_udata > 0)
        
        N1_udata = round(5 * N_udata / M);
        
        N1_udata = min(N1_udata,size(udata,1));
        
        
        du = randi(N_udata,N1_udata,1);
        
        udata1 = udata(du,:);
        
        uhs1 = uhs(du);
        
    else
        
        udata1 = [];
        
        uhs1 = [];
        
    end
    
    %udata1 = udata(d,:)
  end
  tree = treeTrain(data1,hs1,udata1,uhs1,prmTree);
  if(i==1), forest=tree(ones(M,1)); else forest(i)=tree; end
end

end

function tree = treeTrain( data, hs, udata,uhs, prmTree )
% Train single random tree.
[H,F1,minCount,minChild,maxDepth,fWts,split,discretize] = deal(prmTree{:});
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
    %[~,order1]=sort(data1); order1=uint32(order1-1);
    
    if(Nu > 0)
        
        datau1 = udata(didus1,fids1);
        
        uhs1 = uhs(didus1);
        
        [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1);
        
        fid=fids1(fid); 
        
        left=data(dids1,fid)<thr; 
        
        count0=nnz(left);
    
        leftu = udata(didus1,fid) < thr;
        
    else
        
        [~,order1] = sort(data1);
        
        order1 = uint32(order1-1);
        
        [fid,thr,gain] = forestFindThr(data1,hs1,single(ones(n1,1) / n1),order1,H,split);
        
        fid = fids1(fid);
        
        left = data(dids1,fid)<thr;
        
        count0=nnz(left);
        
        leftu = [];
        
    end
    
    
    if( gain>1e-10 && count0>=minChild && (n1-count0)>=minChild )
        
        child(k)=K;
        
        fids(k)=fid-1;
        
        thrs(k)=thr;
        dids{K}=dids1(left);
        
        dids{K+1}=dids1(~left);
        
        didus{K} = didus1(leftu); 
        
        didus{K + 1} = didus1(~leftu);
        
        depth(K:K+1)=depth(k)+1; 
        
        K=K+2;
    end; k=k+1;
  
end
% create output model struct
K=1:K-1; if(discr), hsn={hsn(K)}; else hsn=[hsn{K}]'; end

tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
    'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K),...
    'countu',countu);

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

function [fid,thr,gain] = SSLforestFindThr(data1,hs1,udata1,uhs1)

n_iter = 10;

Nu = size(udata1,1);

Nl = size(data1,1);

Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);


gini_gain = zeros(Nu,size(data1,2));

gini_initial = zeros(size(data1,2),1);

[~,order1]=sort(data1);

order1 = uint32(order1-1);


% save('tmp_error_sav1.mat','data1','hs1');

[fid,thr,gain] = forestFindThr(data1,hs1,ones(size(hs1),'single') / Nl,order1,2,0);

if(~isempty(udata1))
    
    [~,uorder1]=sort(udata1);
    
    uorder1 = uint32(uorder1-1);
    
%     save('tmp_error_sav2.mat','data1','hs1','udata1','fid');
    
    
    uhs_est = KDE2label(data1(:,fid),hs1,udata1(:,fid));
    
    for iter = 1 : n_iter
        
        pure = all(uhs_est(1) == uhs_est);
        
        if(pure)
            
            break;
            
        end
        
        fid_prev = fid;
        
%         save('tmp_error_sav3.mat','udata1','uhs_est','uorder1');
        
        [fid,thr,gain] = forestFindThr(udata1,uhs_est,ones(size(uhs_est),'single') / Nu,uorder1,2,0);
        
        if(fid_prev == fid)
            
            break;
            
        end
        
%         save('tmp_error_sav4.mat','data1','hs1','udata1','fid');
        
        uhs_est = KDE2label(data1(:,fid),hs1,udata1(:,fid));
        
    end
    
    
end

end



function gini_v = approximate_gini_v(hs1)

Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);

Nl = Nl1 + Nl2;

p1 = Nl1 / (Nl + eps);

p2 = Nl2 / (Nl + eps);

gini_v = sqrt(p1 .* p2 / Nl + 1 ./ (max(Nl,1) ^ 2 ));


end


function hs_est = KDE2label(dl,hs1,du)


% Xmin = min(du);
% 
% Xmax = max(du);
% 
% X = Xmin : (Xmax - Xmin) / 100 : Xmax; 


Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);

Nl = length(hs1);

kl1 = kde(double(dl(hs1 == 1)'),'rot');

pkdel1 = evaluate(kl1,double(du)');


kl2 = kde(double(dl(hs1 == 2)'),'rot');

pkdel2 = evaluate(kl2,double(du)');


[~,hs_est] = max([pkdel1;pkdel2]); 

% pkdel1 = pkdel1 * Nl1 / Nl;
% 
% pkdel2 = pkdel2 * Nl2 / Nl;
% 
% p1kde = pkdel1 ./ (pkdel1 + pkdel2 + eps);
% 
% pkdel = pkdel2 + pkdel2;
% 
% pkz_mask = pkdel * Nl < 0.1;
% 
% p1kde(pkz_mask) = 0.5;
% 
% p1x = p1kde(du);


% now assign the labels according to the prior probability


% [~,hs_est] = min([rand(size(du)),p1x'],[],2);

hs_est = uint32(hs_est);


% hs_kde = ones(size(X),'uint32');
% 
% if(Nl2 <  Nl1)
%     
%     hs_kde = hs_kde * 2;
%     
% end
% 
% hs_kde(pkdel1 > pkdel2) = 1;
% 
% hs_kde(pkdel1 < pkdel2) = 2;
% 
% hs_unlabel = hs_kde(du_d);

end


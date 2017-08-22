function forest = SSL_forestTrain_manifold( data, hs, varargin )
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
  'minChild',1, 'maxDepth',64, 'dWts',[], 'fWts',[], 'discretize','','SSL',0,...
  'record',0,'optimal',0,'expSSL',0,'udata',[],'uhs',[],'img_idxs',[],'addInfo',[]};
[M,H,N1,F1,splitStr,minCount,minChild,maxDepth,dWts,fWts,discretize,SSL,...
    record,optimal,expSSL,udata,uhs,img_idxs,addInfo] = ...
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
    record,optimal,expSSL,img_idxs,addInfo};

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
[H,F1,minCount,minChild,maxDepth,fWts,split,discretize,SSL,...
    record,optimal,expSSL,img_idxs,addInfo]=deal(prmTree{:});
N=size(data,1); K=2*N-1; discr=~isempty(discretize);
thrs=zeros(K,1,'single'); distr=zeros(K,H,'single');
fids=zeros(K,1,'uint32'); child=fids; count=fids; depth=fids;




countu = count;

hsn=cell(K,1); dids=cell(K,1); dids{1}=uint32(1:N); k=1; K=2;

Nu = size(udata,1);

didus = cell(K,1);

didus{1} = uint32(1:Nu);


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

if(tree_mode == 11)
    
    gain_u2l = thrs;
    
end




while( k < K )
    % get node data and store distribution
    dids1=dids{k}; dids{k}=[]; hs1=hs(dids1); n1=length(hs1); count(k)=n1;
    
    didus1 = didus{k}; didus{k}=[]; nu1 = length(didus1); countu(k) = nu1;
    
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
    
    if(0)
        
        img_idxs1 = img_idxs(didus1);
        
        
        addInfo.fids1 = fids1;
        
        addInfo.img_idxs = img_idxs1;
        
        addInfo.dids1 = dids1;
        
        addInfo.didus1 = didus1;
        
    end
    
    switch tree_mode
        
        case 11
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,tree_mode);
            
        case 13
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,tree_mode);
            
       
        case 14
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,tree_mode);     
            
            
        case 18
            %
            %             if(length(hs1) > 50)
            %
            %                 [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,1);
            %
            %                 gains_cmp(k,:) = [0 0 0];
            %
            %             else
            
            [fid,thr,gain,gains_cmp(k,:)] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,tree_mode);
            
%             end
            
        otherwise
            
            [fid,thr,gain] = SSLforestFindThr(data1,hs1,datau1,uhs1,addInfo,tree_mode);
            
    end
        
    if(record)
    
        datal{k} = data1;
        
        hsl{k} = hs1;
        
        datau{k} = datau1;
        
        hsu{k} = uhs1;
        
        data_idxs{k} = img_idxs1;
        
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
% 
% if(debug_mode)
%     
%     tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
%         'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K),'gain_l',gain_l,...
%         'gain_w1',gain_w1,'gain_w2',gain_w2);
%     
% else
%     

switch tree_mode
    
    case 11

        tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
            'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K),...
            'gain_u2l',gain_u2l(K),'countu',countu(K));
        
    case 18
        
        tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
            'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K),...
            'countu',countu(K),'gains_cmp',gains_cmp);
 
        
    otherwise
        
        tree=struct('fids',fids(K),'thrs',thrs(K),'child',child(K),...
            'distr',distr(K,:),'hs',hsn,'count',count(K),'depth',depth(K));

        
end


% end


if(record)
    
   tree.datau = datau;
   
   tree.hsu = hsu;
   
   tree.datal = datal;
   
   tree.hsl = hsl;
   
   tree.data_idxs = data_idxs;
    
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

function [fid,thr,gain,gain_cmp] = SSLforestFindThr(data1,hs1,udata1,uhs1,addInfo,tree_mode)


% img_idxs = addInfo.img_idxs;

gain_cmp = [];

if(nargin < 6)
    
   tree_mode = 1; 
    
end

Nl1 = sum(hs1 == 1);

Nl2 = sum(hs1 == 2);


Nu1 = sum(uhs1 == 1);

Nu2 = sum(uhs1 == 2);

g_gain = zeros(size(data1,2),1);

thr_c = zeros(size(data1,2),1);

N = size(udata1,1);

[data1_order,order1] = sort(data1);

nftrs = size(data1,2);

gain_l = zeros(nftrs,1);


thr_l = gain_l;

fid_l = 1 : nftrs;


for ifid = 1 : nftrs
    
    [~,thr_l(ifid),gain_l(ifid)] = forestFindThr(data1(:,ifid),hs1,ones(size(hs1),'single'),uint32(order1(:,ifid)-1),2,0);
    
end

% [fid_bl,thr_bl,gain_bl]=forestFindThr(data1,hs1,ones(size(hs1),'single'),uint32(order1-1),2,0);



switch tree_mode
    
   case 21
      
       h_gku = 1.06 * sqrt(mean(udata1 .^ 2)) * (size(udata1,1) ^ (-0.2));
       
       gain_l2 = evaluate_gini_points_dist(data1,...
           hs1,1 : nftrs,thr_l,h_gku);
       
       [gain,fid] = max(gain_l2);
       
       thr = thr_l(fid);

    
       gain_u = evaluate_argument_data([udata1;data1],...
           [uhs1;hs1],1 : nftrs,thr_l);
       
       [~,fids] = sort(gain_l2,'descend');
       
       fids = fids(1:3);
       
       [~,fid] = max(gain_u(fids));
       
       fid = fids(fid);
       
       gain = gain_l2(fid);
       
    
%        [gain,fid] = max(gain_l2);
       
       thr = thr_l(fid);
       
       
       
%        [gain_u(fid),max(gain_u)]
%            
%        [gain_u(fid),max(gain_u)];

%       
%       plot(1:nftrs,gain_l2,'r',1:nftrs,gain_l1,'g',1:nftrs,gain_u,'b--');
      
      if(0)
          
          plot(1:nftrs,gain_l2,'r',1:nftrs,gain_u,'b--');
          
          
          
          du = udata1(:,fid);
          
          
          dh_distr = histc(du,0:256);
          
          dh_distr(1) = 0;
          
          dh_distr(end) = 0;
          
          
          %             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
          
          
          dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
          
          dh1_distr(1) = 0;
          
          dh1_distr(end) = 0;
          
          dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
          
          dh2_distr(1) = 0;
          
          dh2_distr(end) = 0;
          
          dl = data1(:,fid);
          
          (sum(dl > 250) + sum(dl < 3)) /  length(dl)
          
          sum(hs1(dl > 250) == 1) / sum(hs1(dl > 250) == 2)

          sum(hs1(dl < 3) == 1) / sum(hs1(dl < 3) == 2)

          
          (sum(du > 250) + sum(du < 3)) /  length(du)

          sum(uhs1(du > 250) == 1) / sum(uhs1(du > 250) == 2)

          sum(uhs1(du < 3) == 1) / sum(uhs1(du < 3) == 2)
          
          
          
%           plot(0:256,dh_distr,'g',dl(hs1 == 1),max(dh1_distr),'r*',dl(hs1 == 2),max(dh2_distr),'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');
          
          
          
          
          
          
          [~,fid] = max(gain_u);
          
          
          dh_distr = histc(du,0:256);
          
          dh_distr(1) = 0;
          
          dh_distr(end) = 0;
          
          
          %             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
          
          
          dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
          
          dh1_distr(1) = 0;
          
          dh1_distr(end) = 0;
          
          dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
          
          dh2_distr(1) = 0;
          
          dh2_distr(end) = 0;
          
          dl = data1(:,fid);
          
          
          (sum(dl > 250) + sum(dl < 3)) /  length(dl)
          
          sum(hs1(dl > 250) == 1) / sum(hs1(dl > 250) == 2)

          sum(hs1(dl < 3) == 1) / sum(hs1(dl < 3) == 2)

          
          (sum(du > 250) + sum(du < 3)) /  length(du)

          sum(uhs1(du > 250) == 1) / sum(uhs1(du > 250) == 2)

          sum(uhs1(du < 3) == 1) / sum(uhs1(du < 3) == 2)
          
%           plot(0:256,dh_distr,'g',dl(hs1 == 1),max(dh1_distr),'r*',dl(hs1 == 2),max(dh2_distr),'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');

          
      end
       
    
    case 19
        
%       [gain,fid] = max(gain_l);
%         
%       thr = thr_l(fid);
      
%       gain_u = evaluate_argument_data([udata1;data1],...
%           [uhs1;hs1],1 : nftrs,thr_l);
% %       
%       gain_l1 = evaluate_argument_data(data1,...
%           hs1,1 : nftrs,thr_l);
      

      
      
      
      h_gku = 1.06 * sqrt(mean(udata1 .^ 2)) * (size(udata1,1) ^ (-0.2));

      gain_l2 = evaluate_gini_points_dist(data1,...
          hs1,1 : nftrs,thr_l,h_gku);
      
      [gain,fid] = max(gain_l2);
        
      thr = thr_l(fid);
%       
%       plot(1:nftrs,gain_l2,'r',1:nftrs,gain_l1,'g',1:nftrs,gain_u,'b--');
%       
%       [~,fid_l] = max(gain_l1);
%       
%       [gain_u(fid_l),gain_u(fid),max(gain_u)]
%       
%       [gain_u(fid_l),gain_u(fid),max(gain_u)];
      
      if(0)
          
          dl = data1(:,fid);
          
          du = udata1(:,fid);
          
          
          dh_distr = histc(du,0:256);
          
          dh_distr(1) = 0;
          
          %             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
          
          
          dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
          
          dh1_distr(1) = 0;
          
          dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
          
          dh2_distr(1) = 0;
          
          dl = data1(:,fid);
          
          plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');
          
          
          [~,fid] = max(gain_u);
          
          
          dl = data1(:,fid);
          
          du = udata1(:,fid);
          
          
          dh_distr = histc(du,0:256);
          
          dh_distr(1) = 0;
          
          %             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
          
          
          dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
          
          dh1_distr(1) = 0;
          
          dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
          
          dh2_distr(1) = 0;
          
          dl = data1(:,fid);
          
          plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');
          
          
      end
      
      
      
    case 20
        
        %       [gain,fid] = max(gain_l);
        %
        %       thr = thr_l(fid);
        
        %       gain_u = evaluate_argument_data([udata1;data1],...
        %           [uhs1;hs1],1 : nftrs,thr_l);
        %
        %       gain_l1 = evaluate_argument_data(data1,...
        %           hs1,1 : nftrs,thr_l);
        
        gain_u = evaluate_argument_data([udata1;data1],...
            [uhs1;hs1],1 : nftrs,thr_l);
        
        [gain,fid] = max(gain_u);
        
        thr = thr_l(fid);
      
      
      
%       plot(1:nftrs,gain_l2,'r',1:nftrs,gain_l1,'g',1:nftrs,gain_u,'b--');
      
      
%      h_gk2 = 1.06 * sqrt(mean(data1 .^ 2)) * (size(data1,1) ^ (-0.2));


    
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
        
        
%         i_img = randi(20);
%         
        imgs_distr = histc(img_idxs,1:20);
        
        [~,most_frequent_img] = max(imgs_distr);
        
        i_img = most_frequent_img;
        
        
        if(v_gini > 0)
           
            %hs_KNN = KNN_label2unlabel(data1,hs1,udata1);
            
            gain_single_img = evaluate_argument_data([udata1(img_idxs == i_img,:);data1],...
                [uhs1(img_idxs == i_img);hs1],1 : nftrs,thr_l);
            
            [~,fid] = max(gain_single_img);
            
            
            if(0)
                
               [~,fidl] = max(gain_l);
               
               gain_u = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
               
%                plot(1:45,gain_u,'g',1:45,gain_l,'r',1:45,gain_single_img,'b');
               
             %  [gain_u(fidl),gain_u(fid),max(gain_u)]
               
                
            end
            
            
            
            
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
        
        thr_l = zeros(nftrs,1);
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        [gain_l1,nll1,nll2,nlr1,nlr2] = evaluate_argument_data(data1,hs1,1 : nftrs,thr_l);
                        
        [gain_u,nul1,nul2,nur1,nur2] = evaluate_argument_data(udata1,uhs1,1 : nftrs,thr_l);
%         
        [gain,fid] = max(gain_l);
        
        [~,fid_u] = max(gain_u);
        
        thr = thr_l(fid);
        
        plot(1:45,gain_l,'r',1:45,gain_u,'g',1:45,gain_u + v_gini,'g--');
        
        [max(gain_u)  gain_u(fid)]
        
        [nll1(fid),nll2(fid),nlr1(fid),nlr2(fid);nul1(fid),nul2(fid),nur1(fid),nur2(fid)]
        
        [nll1(fid_u),nll2(fid_u),nlr1(fid_u),nlr2(fid_u);nul1(fid_u),nul2(fid_u),nur1(fid_u),nur2(fid_u)]
        
        [max(gain_u)  gain_u(fid)];
        
        
    case 12

        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
        imgs_distr = histc(img_idxs,1:20);
        
        [~,most_frequent_img] = max(imgs_distr);
        
        i_img = most_frequent_img;
        
        
        
        [gain_single_img,nll1,nll2,nlr1,nlr2] = evaluate_argument_data...
            (udata1(img_idxs == i_img,:),uhs1(img_idxs == i_img),1 : nftrs,thr_l);
        
        [~,fid] = max(gain_single_img);
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        
        
    case 13
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
        imgs_distr = histc(img_idxs,1:20);
        
        [~,most_frequent_img] = max(imgs_distr);
        
        i_img = most_frequent_img;
        
        
        
        [gain_single_img,nsl1,nsl2,nsr1,nsr2] = evaluate_argument_data...
            (udata1(img_idxs == i_img,:),uhs1(img_idxs == i_img),1 : nftrs,thr_l);
        
        [gain_l1,nll1,nll2,nlr1,nlr2] = evaluate_argument_data...
            (data1,hs1,1 : nftrs,thr_l);
        
        [gain_u,nul1,nul2,nur1,nur2] = evaluate_argument_data...
            (udata1,uhs1,1 : nftrs,thr_l);
        
        
        [~,fid] = max(gain_single_img);
        
        [~,fid_l] = max(gain_l);
        
        [gain_u(fid_l), gain_u(fid), max(gain_u)]
        
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        if((gain_u(fid) - gain_u(fid_l)) > 0.01)
            
            
            [~,fid_u] = max(gain_u);
            
%             [max(gain_u)  gain_u(fid)]

            [nll1(fid_l),nll2(fid_l),nlr1(fid_l),nlr2(fid_l);nul1(fid_l),nul2(fid_l),nur1(fid_l),nur2(fid_l);...
                nll1(fid),nll2(fid),nlr1(fid),nlr2(fid);nul1(fid),nul2(fid),nur1(fid),nur2(fid);...
                nll1(fid_u),nll2(fid_u),nlr1(fid_u),nlr2(fid_u);nul1(fid_u),nul2(fid_u),nur1(fid_u),nur2(fid_u)]
            
            img_fn = addInfo.train_imgs_list{i_img};
            
            img = imread(img_fn);
            
            fids = addInfo.fids1;
            
            dataxy = addInfo.train_idxs;
            
            dataxy = dataxy(addInfo.didus1);
            
            
                        dl = data1(:,fid_l);
            
            dl_distr = histc(dl,0:25:256);
            
            dh_distr = histc(udata1(:,fid_l),0:256);
            
%             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
            
            
            dh1_distr = histc(udata1(uhs1 == 1,fid_l),0:256);
            
            dh2_distr = histc(udata1(uhs1 == 2,fid_l),0:256);
            
            
            plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid_l),max(dh_distr),'mo');
            
            
            dl = data1(:,fid);
            
            dl_distr = histc(dl,0:25:256);
            
            dh_distr = histc(udata1(:,fid),0:256);
            
            dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
            
            dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
            
            plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');

            
            du = udata1(:,fid);
            
           thr_l(fid_l)

        end

        
    case 14
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
        [gain_r2,MISE_r1] = evaluate_unlabel_KDE...
            (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,1);
        
        
        if(0)
            
            imgs_distr = histc(img_idxs,1:20);
            
            [~,most_frequent_img] = max(imgs_distr);
            
            i_img = most_frequent_img;
            
            
            
            [gain_single_img,nsl1,nsl2,nsr1,nsr2] = evaluate_argument_data...
                (udata1(img_idxs == i_img,:),uhs1(img_idxs == i_img),1 : nftrs,thr_l);
            
            [gain_l1,nll1,nll2,nlr1,nlr2] = evaluate_argument_data...
                (data1,hs1,1 : nftrs,thr_l);
            
            [gain_u,nul1,nul2,nur1,nur2] = evaluate_argument_data...
                (udata1,uhs1,1 : nftrs,thr_l);
            
            
        end
        
        
        [~,fid] = max(gain_r2);
        
        [~,fid_l] = max(gain_l);
        
%         [gain_u(fid_l), gain_u(fid)]
        
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        
        if(0)
            
            
            [~,fid_u] = max(gain_u);
            
            img_fn = addInfo.train_imgs_list{i_img};
            
            img = imread(img_fn);
            
            fids = addInfo.fids1;
            
            dataxy = addInfo.train_idxs;
            
            dataxy = dataxy(addInfo.didus1);
            
            
            dl = data1(:,fid_l);
            
            du = udata1(:,fid_l);
            
            Nu = length(du);
            
           
            dh_distr = histc(udata1(:,fid_l),0:256);
            
%             plot(0:256,dh_distr,dl(hs1 == 1),1,'r*',dl(hs1 == 2),1,'b*')
            
            
            dh1_distr = histc(udata1(uhs1 == 1,fid_l),0:256);
            
            dh2_distr = histc(udata1(uhs1 == 2,fid_l),0:256);
            
            
            
            bwu = 1.06 * std(udata1(:,fid_l)) * size(udata1,1) ^ (-0.2);
            
            bwu = double(bwu);
            
%             bwl = 1.06 * std(udata1(:,fid_l)) * size(data1,1) ^ (-0.2);
            
            
            
            X = 0:256;
            
            kl = kde(double(dl)',bwu);
            
            pel = evaluate(kl,X) * length(du);
            
            
            kl = kde(double(dl)',bwu);
            
            pel = evaluate(kl,X) * length(du);
            
            
            kl1 = kde(double(dl(hs1 == 1)'),bwu);
            
            pel1 = evaluate(kl1,X);
            
            pel1 = pel1 * length(du) * sum(hs1 == 1) / length(hs1);

            
            kl2 = kde(double(dl(hs1 == 2)'),bwu);
            
            pel2 = evaluate(kl2,X); 
            
            pel2 = pel2 * length(du) * sum(hs1 == 2) / length(hs1);

            
            
            ku = kde(double(du)',bwu);
            
            peu = evaluate(ku,X);
            
            
            subplot(4,1,1);
            
            plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),max(dh_distr),...
                'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid_l),max(dh_distr),'mo');
            
            subplot(4,1,2);
            
            plot(X,pel,'k',X,dh_distr,'g',X,pel1,'r',X,pel2,'b');
          
            
            
%             dl_distr = histc(dl,0:25:256);
            
            dh_distr = histc(udata1(:,fid),0:256);
            
            
            dh1_distr = histc(udata1(uhs1 == 1,fid),0:256);
            
            dh2_distr = histc(udata1(uhs1 == 2,fid),0:256);
            
            dl = data1(:,fid);

            du = udata1(:,fid);
            
            
            kl = kde(double(dl)',bwu);
            
            pel = evaluate(kl,X) * length(du);
            
            
            kl1 = kde(double(dl(hs1 == 1)'),bwu);
            
            pel1 = evaluate(kl1,X);
            
            pel1 = pel1 * length(du) * sum(hs1 == 1) / length(hs1);
            
            
            kl2 = kde(double(dl(hs1 == 2)'),bwu);
            
            pel2 = evaluate(kl2,X);
            
            pel2 = pel2 * length(du) * sum(hs1 == 2) / length(hs1);

            subplot(4,1,3);
            
            plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),max(dh_distr),'b*',...
                0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr_l(fid),max(dh_distr),'mo');

     
            subplot(4,1,4);
            
            plot(X,pel,'k',X,dh_distr,'g',X,pel1,'r',X,pel2,'b');
            
            
  
            
            [gain_r1,MISE_r1] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,1);
            
            [gain_r2,MISE_r2] = evaluate_unlabel_KDE...
                 (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,3);

            
            
            [gain_l(fid_l), gain_l(fid);gain_r1(fid_l), gain_r1(fid);...
                gain_r2(fid_l), gain_r2(fid)]
            
            [MISE_r1(fid_l)  MISE_r1(fid)]
            
            [gain_u(fid_l), gain_u(fid)]

            
        end
        
        
    case 15
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
        
        
        [gain_r2,MISE_r1] = evaluate_unlabel_KDE...
            (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,2);
        
        
        
        [~,fid] = max(gain_r2);
        
%         [~,fid_l] = max(gain_l);
%         
        thr = thr_l(fid);
        
        gain = gain_l(fid);
%         
%         
%         [gain_u,nul1,nul2,nur1,nur2] = evaluate_argument_data...
%             (udata1,uhs1,1 : nftrs,thr_l);
%         
%         
%         [~,fid_u] = max(gain_u);
        
        
%         [gain_u(fid_l) gain_u(fid) max(gain_u)]
%         
%         [gain_u(fid_l) gain_u(fid) max(gain_u)];
        
        
        
        if(0)
            
            [gain_r1,MISE_r1] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,1);
            
            [gain_r2,MISE_r2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,2);
            
%             [gain_r2,MISE_r2] = evaluate_unlabel_KDE...
%                  (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,3);
             
             
            [gain_u(fid_l), gain_u(fid), gain_u(fid_u)]
             
             
            illu_KDE(data1(:,fid_l),udata1(:,fid_l),hs1,uhs1,thr_l(fid_l));

            illu_KDE(data1(:,fid),udata1(:,fid),hs1,uhs1,thr_l(fid));
            
            illu_KDE(data1(:,fid_u),udata1(:,fid_u),hs1,uhs1,thr_l(fid_u));
            
            
            
            
            [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid_u,thr_l(fid_u),2);
            
            illu_PEL(pel1,pel2,data1(:,fid_u),udata1(:,fid_u),hs1,...
                uhs1,thr_l(fid_u));

            
            [gain_l(fid_l), gain_l(fid), gain_l(fid_u);gain_r1(fid_l),...
                gain_r1(fid), gain_r1(fid_u);...
                gain_r2(fid_l), gain_r2(fid), gain_r2(fid_u)]
            
            [MISE_r1(fid_l), MISE_r1(fid) , MISE_r1(fid_u)]
            
            
            [gain_u(fid_l), gain_u(fid), gain_u(fid_u)]
            
   
            
        end
        
        
        
    case 16
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        [gain_r2,MISE_r1] = evaluate_unlabel_KDE...
            (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,2);
        
        
        gain_u = evaluate_argument_data...
            (udata1,uhs1,1 : nftrs,thr_l);
        
        [~,fid] = max(gain_r2);
        
        [~,fid_l] = max(gain_l);
        
%         [gain_u(fid_l), gain_u(fid)]
        
        
        thr = thr_l(fid);
        
        gain = gain_l(fid);
        
        if((max(gain_u) - gain_u(fid)) > 0.02)
            
            [~,fid_u] = max(gain_u);
            
            [gain_u(fid_l), gain_u(fid), gain_u(fid_u);...            
            gain_l(fid_l), gain_l(fid), gain_l(fid_u);...
            gain_r2(fid_l), gain_r2(fid), gain_r2(fid_u)]
             
%              
%             illu_KDE(data1(:,fid_l),udata1(:,fid_l),hs1,uhs1,thr_l(fid_l));
% 
%             illu_KDE(data1(:,fid),udata1(:,fid),hs1,uhs1,thr_l(fid));
%             
%             illu_KDE(data1(:,fid_u),udata1(:,fid_u),hs1,uhs1,thr_l(fid_u));
            
            
            if(gain_u(fid_l) > gain_u(fid))
                
                [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                    (data1,hs1,udata1,uhs1,fid,thr_l(fid_l),2);
                
                illu_PEL(pel1,pel2,data1(:,fid_l),udata1(:,fid_l),hs1,...
                    uhs1,thr_l(fid_l));
                
            end
            
            
            
            [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid,thr_l(fid),2);
            
            illu_PEL(pel1,pel2,data1(:,fid),udata1(:,fid),hs1,...
                uhs1,thr_l(fid));
            
            
            [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid_u,thr_l(fid_u),2);
            
            illu_PEL(pel1,pel2,data1(:,fid_u),udata1(:,fid_u),hs1,...
                uhs1,thr_l(fid_u));
            
            
        end
        
        
    case 17
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
        ehs = KNN_label2unlabel(data1,hs1,udata1);
        
        
        [gain_r2,MISE_r1] = evaluate_unlabel_KDE...
            (data1,hs1,udata1,ehs,1 : nftrs,thr_l,4);
        
        
%         gain_u = evaluate_argument_data...
%             (udata1,uhs1,1 : nftrs,thr_l);
        
        [~,fid] = max(gain_r2);
        
%         [~,fid_l] = max(gain_l);
        
%         [gain_u(fid_l), gain_u(fid)]
        
        
        thr = thr_l(fid);
        
        gain = gain_r2(fid);
        
        if(0)
            
            [~,fid_u] = max(gain_u);
            
            [gain_u(fid_l), gain_u(fid), gain_u(fid_u);...            
            gain_l(fid_l), gain_l(fid), gain_l(fid_u);...
            gain_r2(fid_l), gain_r2(fid), gain_r2(fid_u)]
             
%              
%             illu_KDE(data1(:,fid_l),udata1(:,fid_l),hs1,uhs1,thr_l(fid_l));
% 
%             illu_KDE(data1(:,fid),udata1(:,fid),hs1,uhs1,thr_l(fid));
%             
%             illu_KDE(data1(:,fid_u),udata1(:,fid_u),hs1,uhs1,thr_l(fid_u));




            if(0)
                
                [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                    (data1,hs1,udata1,uhs1,fid,thr_l(fid),4);
                
                illu_PEL(pel1,pel2,data1(:,fid),udata1(:,fid),hs1,...
                    uhs1,thr_l(fid));
                
            end
            
            
            
            [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid,thr_l(fid),4);
            
            illu_PEL(pel1,pel2,data1(:,fid),udata1(:,fid),hs1,...
                uhs1,thr_l(fid));
            
            
            [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid_u,thr_l(fid_u),4);
            
            illu_PEL(pel1,pel2,data1(:,fid_u),udata1(:,fid_u),hs1,...
                uhs1,thr_l(fid_u));
            
            
        end
        
        
    case 18
        
        % the threshold selection
        
        [gain_l,thr_l_idx] = min(gini,[],2);
        
        thr_l_idx = min(thr_l_idx + 1,size(data1,1));
        
        thr_l_idx = sub2ind(size(data1),thr_l_idx,(1:nftrs)');
        
        thr_l = data1_order(thr_l_idx);
        
        v_gini = approximate_gini_v(hs1);
        
        gain_l = gini_initial - gain_l;
        
        
%         [gain_r1,MISE_r1] = evaluate_unlabel_KDE...
%             (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,2);        
for ir =  1 : 10
    
    [gain_r2_tmp(ir,:)] = evaluate_unlabel_KDE...
        (data1,hs1,udata1,uhs1,1 : nftrs,thr_l,4);
    
    
end

        gain_r2 = mean(gain_r2_tmp);
        
        

        
        
        gain_u = evaluate_argument_data...
            (udata1,uhs1,1 : nftrs,thr_l);
        
        [~,fid] = max(gain_r2);
        
        [~,fid_l] = max(gain_l);
        
        
        gain_cmp = [gain_u(fid_l) gain_u(fid) max(gain_u)];
        
        
        
        
%         [gain_u(fid_l), gain_u(fid)]
        
        
        thr = thr_l(fid);
        
        gain = gain_r2(fid);
        
        
        if((gain_u(fid_l) - gain_u(fid)) > 0.01)
            
            [~,fid_u] = max(gain_u);
            
            [gain_u(fid_l), gain_u(fid), gain_u(fid_u);...
                gain_l(fid_l), gain_l(fid), gain_l(fid_u);...
                gain_r2(fid_l), gain_r2(fid), gain_r2(fid_u)]
            
            
            if(gain_u(fid_l) > gain_u(fid))
                
                [~,~,pel1,pel2] = evaluate_unlabel_KDE...
                    (data1,hs1,udata1,uhs1,fid,thr_l(fid_l),4);
                
                illu_PEL(pel1,pel2,data1(:,fid_l),udata1(:,fid_l),hs1,...
                    uhs1,thr_l(fid_l));
                
            end
            
            
            
            [tmp_gain1,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid,thr_l(fid),4);
            
            illu_PEL(pel1,pel2,data1(:,fid),udata1(:,fid),hs1,...
                uhs1,thr_l(fid));
            
            
            [tmp_gain2,~,pel1,pel2] = evaluate_unlabel_KDE...
                (data1,hs1,udata1,uhs1,fid_u,thr_l(fid_u),4);
            
            illu_PEL(pel1,pel2,data1(:,fid_u),udata1(:,fid_u),hs1,...
                uhs1,thr_l(fid_u));
            
            
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

function illu_KDE(dl,du,hs1,uhs1,thr)

X = 0:256;


Nu = length(du);

Nl = length(dl);

dh_distr = histc(du,X);

dh1_distr = histc(du(uhs1 == 1),X);

dh2_distr = histc(du(uhs1 == 2),X);

bwu = 1.06 * std(du) * Nu ^ (-0.2);

bwu = double(bwu);


kl = kde(double(dl)',bwu);

pel = evaluate(kl,X) * length(du);



kl1 = kde(double(dl(hs1 == 1)'),bwu);

pel1 = evaluate(kl1,X);

pel1 = pel1 * length(du) * sum(hs1 == 1) / Nl;


kl2 = kde(double(dl(hs1 == 2)'),bwu);

pel2 = evaluate(kl2,X);

pel2 = pel2 * length(du) * sum(hs1 == 2) / length(hs1);


subplot(2,1,1);

plot(0:256,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),max(dh_distr),...
    'b*',0:256,dh1_distr,'r',0:256,dh2_distr,'b', thr,0.618 * max(dh_distr),'ko');

subplot(2,1,2);

plot(X,pel,'k',X,dh_distr,'g',X,pel1,'r',X,pel2,'b',dl(hs1 == 1),1,'r*',...
    dl(hs1 == 2),max(dh_distr),'b*');


end



function illu_PEL(pel1,pel2,dl,du,hs1,uhs1,thr)

X = 0:255;


Nu = length(du);

Nl = length(dl);

dh_distr = histc(du,X);

dh1_distr = histc(du(uhs1 == 1),X);

dh2_distr = histc(du(uhs1 == 2),X);


pel = pel1 + pel2;

bwu = 1.06 * std(du) * Nu ^ (-0.2);

bwu = double(bwu);

kl1 = kde(double(dl(hs1 == 1)'),bwu);

pel1_prev = evaluate(kl1,X);

pel1_prev = pel1_prev * length(du) * sum(hs1 == 1) / Nl;


kl2 = kde(double(dl(hs1 == 2)'),bwu);

pel2_prev = evaluate(kl2,X);

pel2_prev = pel2_prev * length(du) * sum(hs1 == 2) / length(hs1);

pel_prev = pel1_prev + pel2_prev;



subplot(3,1,1);

plot(X,dh_distr,'g',dl(hs1 == 1),1,'r*',dl(hs1 == 2),max(dh_distr),...
    'b*',X,dh1_distr,'r',X,dh2_distr,'b', thr,0.618 * max(dh_distr),'ko');



subplot(3,1,2);

plot(X,dh_distr,'g',X,pel_prev,'k',dl(hs1 == 1),1,'r*',dl(hs1 == 2),max(dh_distr),...
    'b*',X,pel1_prev,'r',X,pel2_prev,'b', thr,0.618 * max(dh_distr),'ko');




subplot(3,1,3);

plot(X,pel,'k',X,dh_distr,'g',X,pel1,'r',X,pel2,'b');


end

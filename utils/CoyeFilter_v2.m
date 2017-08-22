function Z = CoyeFilter_v2(img)


if(size(img,3) > 1)
    
    % Read image
    img = im2double(img);
    % Convert RGB to Gray via PCA
    lab = rgb2lab(img);
    f = 0;
    wlab = reshape(bsxfun(@times,cat(3,1-f,f/2,f/2),lab),[],3);
    
    [C,~,S] = myPCA(wlab,3);
    
    S = reshape(S,size(lab));
    
    S = S(:,:,1);

else
    
    S = double(img);
    
end

if((max(S(:))-min(S(:))) > 0.1)
    
    gray = (S-min(S(:)))./(max(S(:))-min(S(:)));
    
    J = adapthisteq(gray,'numTiles',[8 8],'nBins',128);
    
    %% Background Exclusion
    % Apply Average Filter
    h = fspecial('average', [9 9]);
    JF = imfilter(J, h);
    
    % Take the difference between the gray image and Average Filter
    Z = imsubtract(JF, J);
    
    Z(Z > 0.1) = 0.1;
    
    Z(Z < -0.1) = -0.1;
    
    Z = (Z - mean2(Z)) / std2(Z);
    
else
    
    Z = zeros(size(gray));
    
end

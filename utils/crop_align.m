function [img,gt_img] = crop_align(img,gt_img)

[gtx,gty,gtz] = ind2sub(size(gt_img),find(gt_img > 0));

% limit the region of interest to the bounding cube defined by the
% groundtruth

xmin = max(min(gtx) - 5,1);

ymin = max(min(gty) - 5,1);

zmin = max(min(gtz) - 5,1);

xmax = min(max(gtx) + 5,size(gt_img,1));

ymax = min(max(gty) + 5,size(gt_img,2));

zmax = min(max(gtz) + 5,size(gt_img,3));

gt_img = gt_img(1:xmax,1:ymax,1:zmax);

sz_img = size(img);

sz_gt = size(gt_img);

sz_img = min([sz_img;sz_gt]);

img = img(xmin:sz_img(1),ymin:sz_img(2),zmin:sz_img(3));

gt_img = gt_img(xmin:sz_img(1),ymin:sz_img(2),zmin:sz_img(3));

gt_img = gt_img > 0;
function features = Gabor_Filter(img)


img = double(img);

scale = 1;

filter_size = 40.*scale;

filter_size_halfed = round((filter_size)/2);

Fs = 0.1:0.1:0.3;

sigmas = [2:2:8].*scale;

thetas=pi/8:pi/8:pi-pi/8;


features = zeros([length(img(:)),numel(sigmas) * numel(thetas) * numel(Fs)]);

ci = 1;

for k = 1:numel(sigmas)
    for j = 1:numel(Fs)
        for i = 1:numel(thetas)
            
            sigma = sigmas(k);
            
            F = Fs(j);
            
            theta = thetas(i);
            
            % setup the Gabor transform
            [x,y]=meshgrid(-filter_size_halfed:filter_size_halfed,-filter_size_halfed:filter_size_halfed);
            
            g_sigma = (1./(2*pi*sigma^2)).*exp(((-1).*(x.^2+y.^2))./(2*sigma.^2));
            
            real_g = g_sigma.*cos((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
            
            im_g = g_sigma.*sin((2*pi*F).*(x.*cos(theta)+y.*sin(theta)));
            
            % perform Gabor transform
            uT =sqrt(conv2(img,real_g,'same').^2+conv2(img,im_g,'same').^2);
            
            % normalize transformed image
            uT = (uT-mean(uT(:)))./std(uT(:));
            
            % append tranformed images to 'features'
            features(:,ci) = uT(:);
            
            ci = ci + 1;
            
        end
    end
end

% std(features)

% szG = size(features);

% features = reshape(features,[prod(szG(1:2)),prod(szG(3:end))]);

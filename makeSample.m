% function data = makeSample(filename, imageSize,netPreproc)
%     % Read image
%     img = imread(filename);
%     if nargin > 2 && ~isempty(netPreproc)
%      img = preprocessWithNet(img, netPreproc);
%     end
%     % --- Handle grayscale images ---
%     if size(img,3) == 1
%         img = repmat(img,1,1,3); % Convert grayscale → RGB
%     end
% 
%     % --- Handle CMYK images ---
%     if size(img,3) == 4
%         img = img(:,:,1:3); % Use first 3 channels as RGB
%     end
% 
%     % Resize + convert to single
%     img = im2single(imresize(img, imageSize));
% 
%     % Convert to HSV
%     hsvImg = rgb2hsv(img);
% 
%     % Convert to grayscale
%     grayImg = rgb2gray(img);
% 
%     % Sobel edge map
%     sobelMap = edge(grayImg, 'sobel');
%     sobelMap = im2single(sobelMap);
%     sobelMap = reshape(sobelMap, [size(sobelMap,1), size(sobelMap,2), 1]);
% 
%     % Laplacian of Gaussian (LoG) edge map
%     lapMap = edge(grayImg, 'log');
%     lapMap = im2single(lapMap);
%     lapMap = reshape(lapMap, [size(lapMap,1), size(lapMap,2), 1]);
% 
%     % Gradient magnitude and direction
%     [Gmag, Gdir] = imgradient(grayImg);
%     Gmag = im2single(mat2gray(Gmag)); % Normalize
%     Gdir = im2single(mat2gray(Gdir));
%     Gmag = reshape(Gmag, [size(Gmag,1), size(Gmag,2), 1]);
%     Gdir = reshape(Gdir, [size(Gdir,1), size(Gdir,2), 1]);
% 
%     % Concatenate all channels (3+3+1+1+1+1 = 10)
%     data = cat(3, img, hsvImg, sobelMap, lapMap, Gmag, Gdir);
% 
%     % --- Safety check ---
%     if size(data,3) ~= 10
%         error(['makeSample error: File ', filename, ...
%                ' produced ', num2str(size(data,3)), ' channels instead of 10']);
%     end
% end
function data = makeSample(filename, imageSize, netPreproc)
    % Read image
    img = imread(filename);

    % Apply preprocessing DNN if provided
    if nargin > 2 && ~isempty(netPreproc)
        img = preprocessWithNet(img, netPreproc, [224 224]); % match preproc input
    end

    % --- rest of your 10-channel stacking code ---
    % Convert to HSV, compute edges, LoG, gradient maps, etc.
    hsvImg = rgb2hsv(im2single(imresize(img, imageSize)));
    grayImg = rgb2gray(im2single(imresize(img, imageSize)));

    sobelMap = im2single(edge(grayImg, 'sobel'));
    lapMap   = im2single(edge(grayImg, 'log'));
    [Gmag, Gdir] = imgradient(grayImg);
    Gmag = im2single(mat2gray(Gmag));
    Gdir = im2single(mat2gray(Gdir));

    data = cat(3, im2single(imresize(img, imageSize)), hsvImg, ...
               reshape(sobelMap, [imageSize 1]), ...
               reshape(lapMap,   [imageSize 1]), ...
               reshape(Gmag,     [imageSize 1]), ...
               reshape(Gdir,     [imageSize 1]));

    % Safety check
    if size(data,3) ~= 10
        error(['makeSample error: File ', filename, ...
               ' produced ', num2str(size(data,3)), ' channels instead of 10']);
    end
end

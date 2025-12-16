function w = computeClassWeights(lbls, classes)
% Inverse-frequency class weights (sum to #classes)
    counts = countcats(lbls);
    counts = double(counts);
    invf   = 1 ./ max(counts,1);
    w      = invf / mean(invf);  % normalize around 1
    % Map to the order of 'classes' if needed
    % (lbls categories usually match 'classes' from categories(imdsTrain.Labels))
end
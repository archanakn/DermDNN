classdef weightedClassificationLayer < nnet.layer.ClassificationLayer
    properties
        ClassWeights
    end
    
    methods
        function layer = weightedClassificationLayer(classWeights, name)
            if nargin < 2
                name = 'weighted_ce';
            end
            layer.Name = name;
            layer.ClassWeights = classWeights;
        end
        
        function loss = forwardLoss(layer, Y, T)
            % Y = predictions (softmax output)
            % T = one-hot encoded targets
            W = layer.ClassWeights(:)';   % ensure row vector
            % Apply weights: sum over batch
            N = size(Y,4); 
            loss = 0;
            for i = 1:N
                ti = T(:,:,:,i);
                yi = Y(:,:,:,i);
                ci = find(ti==1);          % true class index
                wi = W(ci);
                loss = loss - wi * log(max(yi(ci),1e-8));
            end
            loss = loss / N;
        end
    end
end

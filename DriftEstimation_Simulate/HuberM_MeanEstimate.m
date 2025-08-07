function robust_mean = HuberM_MeanEstimate(data, delta)
    if nargin < 2
        delta = 1.0; 
    end

    max_iter = 1000; 
    tol = 1e-6; 
    
    mu = median(data);
    
    for iter = 1:max_iter
        residuals = data - mu;
       
        loss = abs(residuals) < delta;
        loss = loss .* (residuals .^ 2 / 2) + ~loss .* (delta * (abs(residuals) - delta / 2));
        
        weights = exp(-loss);
        weighted_mean = sum(weights .* data) / sum(weights);
        
        if abs(weighted_mean - mu) < tol
            break;
        end
        
        mu = weighted_mean;
    end
    
    robust_mean = mu;
end

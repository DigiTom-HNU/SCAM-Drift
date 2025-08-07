function robust_mean = HuberM_MeanEstimate(data, delta)
    if nargin < 2
        delta = 1.0; % 默认 Huber 函数的 delta 值
    end
    % 输入：
    % data - 包含异常数据的数组
    % delta - Huber损失函数的阈值
    
    % 初始化
    max_iter = 1000; % 最大迭代次数
    tol = 1e-6; % 收敛阈值
    
    % 初始估计
    mu = median(data);
    
    for iter = 1:max_iter
        % 计算残差
        residuals = data - mu;
        
        % 计算Huber损失函数
        loss = abs(residuals) < delta;
        loss = loss .* (residuals .^ 2 / 2) + ~loss .* (delta * (abs(residuals) - delta / 2));
        
        
        % 计算加权平均
        weights = exp(-loss);
        weighted_mean = sum(weights .* data) / sum(weights);
        
        % 检查收敛
        if abs(weighted_mean - mu) < tol
            break;
        end
        
        mu = weighted_mean;
    end
    
    % 返回Huber均值
    robust_mean = mu;

%     fprintf('Huber 最小二乘估计稳健均值：%.4f\n', robust_mean);
end

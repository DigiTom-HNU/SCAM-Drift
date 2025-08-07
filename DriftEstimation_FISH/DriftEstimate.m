function [DriftX_diff,DriftY_diff,DriftZ_diff] = DriftEstimate(locTable_in,frameNum,dimensional,colrow)
[uniqueVals, indices, ~] = unique(locTable_in(:, 1));
[counts, ~] = histcounts(locTable_in(:, 1), [uniqueVals;uniqueVals(end)+1]);
diffDrift = [];
for i = 1:length(indices)
    eachClass = indices(i);
    classInfo = locTable_in(eachClass:eachClass+counts(i)-1, :);
    diffDriftXT = [0;diff(classInfo(:,3))];
    diffDriftYT = [0;diff(classInfo(:,4))];
    if strcmp(dimensional, '3D')
        diffDriftZT = [0;diff(classInfo(:,5))];
        diffDrift = [diffDrift;[classInfo(:,2),diffDriftXT,diffDriftYT,diffDriftZT]];
    else
        diffDrift = [diffDrift;[classInfo(:,2),diffDriftXT,diffDriftYT]];
    end
end
diffDriftNo1F = diffDrift(diffDrift(:,1)~=1,:);
if strcmp(dimensional, '3D')
    non_zero_rows = any(diffDriftNo1F(:, 2:4), 2);
else
    non_zero_rows = any(diffDriftNo1F(:, 2:3), 2);
end
diffDriftNo1F = diffDriftNo1F(non_zero_rows, :);
diffDrift = [diffDrift(diffDrift(:,1)==1,:);diffDriftNo1F];
%% 符合正态分布时效果好，聚类效果不好就不符合，丢了很多信息
% 一帧的漂移差异数量少时就是平均分布
% n = 10;
% diffDriftT = diffDrift;
% diffDriftT(:,2:3) = diffDriftT(:,2:3)*163.8;
% diffDriftXframe  = diffDriftT(diffDriftT(:,1)==n,2);
% 
% [counts, edges] = histcounts(diffDriftXframe);
% bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
% [fitResult,~] = fit(bin_centers', counts', 'gauss1');
% figure;histogram(diffDriftXframe, 'Normalization', 'count');
% hold on;
% fit_values = feval(fitResult, bin_centers);
% plot(bin_centers, fit_values, 'r-', 'LineWidth', 2);  % 绘制拟合曲线
% xlabel('X drift between frames (nm)');
% ylabel('Counts');
% set(gca,'fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2);
% 
% diffDriftYframe  = diffDriftT(diffDriftT(:,1)==n,3);
% figure;histogram(diffDriftYframe);
% xlabel('Y drift between frames (nm)');
% ylabel('Counts');
% set(gca,'fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2);
% if strcmp(dimensional, '3D')
%     diffDriftZframe  = diffDriftT(diffDriftT(:,1)==n,4);
%     figure;histogram(diffDriftZframe);
%     xlabel('Z drift between frames (nm)');
%     ylabel('Counts');
%     set(gca,'fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2);
% end
% value_counts = accumarray(diffDrift(:, 1), 1);
% [counts, edges] = histcounts(value_counts);
% bin_centers = (edges(1:end-1) + edges(2:end)) / 2;
% [fitResult,~] = fit(bin_centers', counts', 'gauss1');
% figure;histogram(value_counts, 'Normalization', 'count');
% hold on;
% fit_values = feval(fitResult, bin_centers);
% plot(bin_centers, fit_values, 'r-', 'LineWidth', 2);  % 绘制拟合曲线
% xlabel('The num of drift per frame');
% ylabel('Counts');
% set(gca,'fontsize',18, 'FontWeight', 'bold', 'FontName', 'Arial','LineWidth', 2);

%% mcmc贝叶斯估计
DriftX_diff = zeros(frameNum,1);
DriftY_diff = zeros(frameNum,1);
DriftZ_diff = zeros(frameNum,1);
diffDrift(:,1) = diffDrift(:,1) - colrow{2}(1) + 1;
parfor i= 1:frameNum
    diffDriftT = diffDrift(diffDrift(:,1) == i,:);
    if isempty(diffDriftT)
        continue;
    end
%     DriftX_diff(i) = mean(diffDriftT(:,2));
%     DriftY_diff(i) = mean(diffDriftT(:,3));
    DriftX_diff(i) = HuberM_MeanEstimate(diffDriftT(:,2));
    DriftY_diff(i) = HuberM_MeanEstimate(diffDriftT(:,3));
%     DriftX_diff(i) = bayesian_estimation_mcmc_adaptive(diffDriftT(:,2));
%     DriftY_diff(i) = bayesian_estimation_mcmc_adaptive(diffDriftT(:,3));
    if strcmp(dimensional, '3D')
%           DriftZ_diff(i) = mean(diffDriftT(:,4));
        DriftZ_diff(i) = HuberM_MeanEstimate(diffDriftT(:,4));
%         DriftZ_diff(i) = bayesian_estimation_mcmc_adaptive(diffDriftT(:,4));
    end
end

% DriftX_diff = accumarray(diffDrift(:, 1), diffDrift(:, 2), [], @mean);
% DriftY_diff = accumarray(diffDrift(:, 1), diffDrift(:, 3), [], @mean);
% if strcmp(dimensional, '3D')
%     DriftZ_diff = accumarray(diffDrift(:, 1), diffDrift(:, 4), [], @mean);
% else
%     DriftZ_diff = [];
% end
%% 剔除异常值求均值 对一帧内位移差多的效果好一些
% [~, ~, unique_indices] = unique(diffDrift(:, 1));
% remove_outliers = @(x) x(~isoutlier(x));
% DriftX_diff = splitapply(@(x) mean(remove_outliers(x)), diffDrift(:, 2), unique_indices);
% DriftY_diff = splitapply(@(x) mean(remove_outliers(x)), diffDrift(:, 3), unique_indices);
% if strcmp(dimensional, '3D')
%     DriftZ_diff = splitapply(@(x) mean(remove_outliers(x)), diffDrift(:, 4), unique_indices);
% else
%     DriftZ_diff = [];
% end
end


function mu_posterior = bayesian_estimation_mcmc_adaptive(obs)
    % 观测数据
    x = obs;

    % 已知的标准差
    sigma = std(x);  % 假设已知的观测数据的标准差

    outliers = isoutlier(x);
    % 自适应计算先验分布参数
    mu0_initial = mean(x(~outliers));  % 使用数据均值作为初始先验均值
    sigma0_initial = 0.0425;  % 使用数据标准差作为初始先验标准差

    % MCMC 参数
    num_samples = 20000; % MCMC样本数
    mu_samples = zeros(num_samples, 1);
    sigma0_samples = zeros(num_samples, 1); % 记录先验标准差的样本
    acceptance_rate_target = 0.25; % 目标接受率
    proposal_sd = sigma0_initial / 2; % 提案分布的初始标准差
    adapt_rate = 0.01; % 自适应调整率

    % 初始值
    mu_current = mu0_initial;
    sigma0_current = sigma0_initial;
    % MCMC 采样
    accept_count = 0;
    
    normalizing_const_mu0 = -0.5 * log(2 * pi * sigma0_initial^2);
    normalizing_const_sigma = -0.5 * log(2 * pi * sigma^2);
    % Metropolis-Hastings 算法
    for i = 1:num_samples
        % 提案分布：正态分布
        mu_proposed = mu_current + proposal_sd * randn;
        sigma0_proposed = sigma0_current + proposal_sd * randn;
        
        % 计算后验概率的对数
        log_posterior_current = -0.5 * ((mu_current - mu0_initial).^2 / sigma0_initial^2) + normalizing_const_mu0 ...
            + sum(-0.5 * ((x - mu_current).^2 / sigma^2) + normalizing_const_sigma);
        log_posterior_proposal = -0.5 * ((mu_proposed - mu0_initial).^2 / sigma0_initial^2) + normalizing_const_mu0 ...
            + sum(-0.5 * ((x - mu_proposed).^2 / sigma^2) + normalizing_const_sigma);
   
        % 计算接受率
        log_alpha = exp(log_posterior_proposal - log_posterior_current);
        
        % 接受或拒绝提案
        if rand < log_alpha
            mu_current = mu_proposed;
            sigma0_current = sigma0_proposed;
            accept_count = accept_count + 1;
        end

        % 存储样本
        mu_samples(i) = mu_current;
        sigma0_samples(i) = sigma0_current;

        % 自适应调整提议分布的标准差
        if mod(i, 100) == 0
            acceptance_rate = accept_count / 100;
            accept_count = 0;
            proposal_sd = proposal_sd * exp(adapt_rate * (acceptance_rate - acceptance_rate_target));
        end
    end

    % 去掉前面的 burn-in 期（例如前 1000 个样本）
    burn_in = 1000;
    mu_samples = mu_samples(burn_in+1:end);

    % 计算后验分布参数
    mu_posterior = mean(mu_samples);

end

% function mu_posterior = bayesian_estimation_mcmc_adaptive(obs)
%     % 观测值
%     x = obs;
% 
%     % 初始估计obs_sigma
%     obs_sigma = std(x);
% 
%     % 设定 MCMC 参数
%     num_samples = 20000; % 采样数量
%     mu_samples = zeros(num_samples, 1);
%     acceptance_rate_target = 0.25; % 目标接受率
%     proposal_sd = 0.0425 / 2; % 初始提议分布标准差
%     adapt_rate = 0.01; % 自适应调整率
% 
%     % 初始值
%     mu_current = mean(x);
% 
%     % MCMC 采样
%     accept_count = 0;
%     for i = 1:num_samples
%         % 生成候选值
%         mu_proposed = mu_current + randn * proposal_sd;
% 
%         % 非信息性先验（广泛的正态分布），忽略先验
%         log_prior_current = 0;
%         log_prior_proposed = 0;
%         log_likelihood_current = -sum((x - mu_current).^2) / (2 * obs_sigma^2);
%         log_likelihood_proposed = -sum((x - mu_proposed).^2) / (2 * obs_sigma^2);
%         
%         log_alpha = (log_prior_proposed + log_likelihood_proposed) - (log_prior_current + log_likelihood_current);
%         
%         % 决定是否接受提议
%         if log(rand) < log_alpha
%             mu_current = mu_proposed;
%             accept_count = accept_count + 1;
%         end
% 
%         % 存储样本
%         mu_samples(i) = mu_current;
% 
%         % 自适应调整提议分布的标准差
%         if mod(i, 100) == 0
%             acceptance_rate = accept_count / 100;
%             accept_count = 0;
%             proposal_sd = proposal_sd * exp(adapt_rate * (acceptance_rate - acceptance_rate_target));
%         end
%     end
% 
%     % 计算后验均值
%     mu_posterior = mean(mu_samples);
% 
%     % 绘制采样结果以进行检查
% %     figure;
% %     subplot(2,1,1);
% %     plot(mu_samples);
% %     title('MCMC Samples');
% %     xlabel('Sample Number');
% %     ylabel('Mu Value');
% % 
% %     subplot(2,1,2);
% %     histogram(mu_samples, 50);
% %     title('Histogram of Mu Samples');
% %     xlabel('Mu Value');
% %     ylabel('Frequency');
% end
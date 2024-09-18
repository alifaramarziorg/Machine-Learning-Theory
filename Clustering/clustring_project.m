
clc
clear
close all

%% Import CSV file to matlab
org_data = readtable('preproccessed data.csv');

%% Data Conversion and Selection
%convert data from table to matrix  
x = org_data{2:end ,:};


%% K-mean

K_values = 2:13;

% 3. Initialize vector to hold SSE values
sse = zeros(size(K_values));

% 4. Run K-means clustering for each K value and compute SSE
for i = 1:length(K_values)
    k = K_values(i);
    [~, centroids, sumd] = kmeans(x, k);
    sse(i) = sum(sumd);
end

% 5. Plot SSE values against K
plot(K_values, sse, 'bo-');
xlabel('Number of Clusters (K)');
ylabel('Sum of Squared Errors (SSE)');
title('Elbow Method for K-means Clustering');
 
%% use PCA
coeff = pca(x);

k = 2;  
x_pca = x * coeff(:, 1:k);

%% apply kmeans with k=4
opts = statset('Display','final');
[idx,C] = kmeans(x,4,'Distance','cityblock',...
    'Replicates',5,'Options',opts);

centers = C * coeff(:, 1:k);

figure;
scatter(x_pca(:, 1), x_pca(:, 2), [], idx, 'filled');
xlabel('X');
ylabel('Y');
title('Clustering Result');
legend('Data Points');
hold on;

scatter(centers(:, 1), centers(:, 2), [], 'rx', 's', 'LineWidth', 2);


legend('Data Points', 'Cluster Centers');
hold off;
%% HAC clustring with single linkage

k_range = 2:13;  
sse_results = zeros(size(k_range));

for i = 1:numel(k_range)
    k = k_range(i);
    
    % Perform Agglomerative Clustering with single linkage
    clustering = linkage(x, 'single');
    
    % Assign cluster labels
    labels = cluster(clustering, 'MaxClust', k);
    
    % Calculate SSE (Sum of Squared Errors)
    centroid = zeros(k, size(x, 2));
    for j = 1:k
        centroid(j, :) = mean(x(labels == j, :));
    end
    sse_results(i) = sum(sum((x - centroid(labels, :)).^2));
end

% Plot SSE results vs. k values
figure ;

plot(k_range, sse_results, 'b-o');
xlabel('Number of Clusters (k)');
ylabel('Sum of Squared Errors (SSE)');
title('SSE vs. Number of Clusters (single linkage)');
grid on;



%% HAC clustring with complete linkage

k_range = 2:13;  
sse_results = zeros(size(k_range));

for i = 1:numel(k_range)
    k = k_range(i);
    
    % Perform Agglomerative Clustering with single linkage
    clustering = linkage(x, 'complete');
    
    % Assign cluster labels
    labels = cluster(clustering, 'MaxClust', k);
    
    % Calculate SSE (Sum of Squared Errors)
    centroid = zeros(k, size(x, 2));
    for j = 1:k
        centroid(j, :) = mean(x(labels == j, :));
    end
    sse_results(i) = sum(sum((x - centroid(labels, :)).^2));
end

% Plot SSE results vs. k values
figure ;
plot(k_range, sse_results, 'b-o');
xlabel('Number of Clusters (k)');
ylabel('Sum of Squared Errors (SSE)');
title('SSE vs. Number of Clusters (complete linkage)');
grid on;
%%
%% HAC clustring with ward linkage

k_range = 2:13;  
sse_results = zeros(size(k_range));

for i = 1:numel(k_range)
    k = k_range(i);
    
    % Perform Agglomerative Clustering with single linkage
    clustering = linkage(x, 'ward');
    
    % Assign cluster labels
    labels = cluster(clustering, 'MaxClust', k);
    
    % Calculate SSE (Sum of Squared Errors)
    centroid = zeros(k, size(x, 2));
    for j = 1:k
        centroid(j, :) = mean(x(labels == j, :));
    end
    sse_results(i) = sum(sum((x - centroid(labels, :)).^2));
end

% Plot SSE results vs. k values
figure ;
plot(k_range, sse_results, 'b-o');
xlabel('Number of Clusters (k)');
ylabel('Sum of Squared Errors (SSE)');
title('SSE vs. Number of Clusters (ward linkage)');
grid on;

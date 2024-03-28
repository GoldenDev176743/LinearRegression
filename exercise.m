data = readtable('linerud.csv');

% Extract features and labels
X = data{:, 1:3}; % First three columns as features
y = data{:, end-2:end}; % Last three columns as labels

% Find NaN values and replace with mean value from data for features
for i = 1:size(X, 2)
    TF = isnan(X(:, i));
    X(TF, i) = 0;
    m = mean(X(:, i));
    X(TF, i) = m;
end

% Find NaN values and replace with mean value from data for labels
for i = 1:size(y, 2)
    TF = isnan(y(:, i));
    y(TF, i) = 0;
    m = mean(y(:, i));
    y(TF, i) = m;
end

% Data transformation
X = zscore(X);
y = zscore(y);

[train_X, test_X, train_y, test_y] = split_data(X, y, 0.2);

k = 1:10;
num_K = length(k);

% Prediction
[euclidean_pred, manhattan_pred] = knn_regression(train_X, train_y, test_X, num_K);

manhattan_mse = zeros(num_K, 1);
euclidean_mse = zeros(num_K, 1);

% Calculating MSE
for i = 1:num_K
    manhattan_mse(i) = immse(manhattan_pred(:, i), test_y);
    euclidean_mse(i) = immse(euclidean_pred(:, i), test_y);
    fprintf('K = %d: Manhattan MSE: %f  Euclidean MSE: %f\n', i, manhattan_mse(i), euclidean_mse(i));
end

% MSE plot
figure
plot(k, manhattan_mse, 'b--o', k, euclidean_mse, 'r--o');
title("Mean Squared Error");
xlabel('k');
ylabel('MSE')

% Data splitting function
function [train_X, test_X, train_y, test_y] = split_data(X, y, test_size)
    rng(42);
    indices = randperm(size(X, 1));
    num_test = ceil(test_size * size(X, 1));

    test_indices = indices(1:num_test);
    train_indices = indices(num_test+1:end);

    train_X = X(train_indices, :);
    test_X = X(test_indices, :);
    train_y = y(train_indices, :);
    test_y = y(test_indices, :);
end

% KNN regression function
function [euclidean_pred, manhattan_pred] = knn_regression(train_X, train_y, test_X, num_K)
    num_test = size(test_X, 1);

    manhattan_pred = zeros(num_test, num_K);
    euclidean_pred = zeros(num_test, num_K);

    for i = 1:num_test
        manhattan_distances = abs(train_X - test_X(i, :)); % Manhattan distance
        euclidean_distances = sqrt(sum((train_X - test_X(i, :)).^2, 2)); % Euclidean distance
        [~, manhattan_sorted_indices] = sort(manhattan_distances);
        [~, euclidean_sorted_indices] = sort(euclidean_distances);
        for j = 1:num_K
            m_k_nearest_indices = manhattan_sorted_indices(1:j);
            e_k_nearest_indices = euclidean_sorted_indices(1:j);
            manhattan_pred(i, j) = mean(train_y(m_k_nearest_indices, :));
            euclidean_pred(i, j) = mean(train_y(e_k_nearest_indices, :));
        end
    end
end

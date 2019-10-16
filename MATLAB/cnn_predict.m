function [results, acc] = cnn_predict(y, Z, model)

addpath(genpath('./cnn'), genpath('./opt'));

prob = check_data(y, Z, model.config);
net = init_net(model);

results = predict(prob, 0.05, model, net);

% Obtain predicted label
[~, results] = max(results, [], 1);
results = results';

% Calculate accuracy
acc = sum(results == y) / length(y);


function [results, acc] = cnn_predict(y, Z, model)

addpath(genpath('./cnn'), genpath('./opt'));

param = model.param;
prob = check_data(y, Z, param);
net = init_net(param, model);

results = predict(prob, param, model, net);

% Obtain predicted label
[~, results] = max(results, [], 1);
results = results';

% Calculate accuracy
acc = sum(results == y) / length(y);


function acc = cal_accuracy(results, y)

% Calculate accuracy
acc = sum(results == y) / length(y);

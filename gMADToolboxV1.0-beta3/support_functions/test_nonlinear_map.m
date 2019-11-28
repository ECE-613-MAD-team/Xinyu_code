function [ yhat ] = test_nonlinear_map( X, beta, ehat, J )
disp('Non-linear mapping...');
model = 'logistic5';
yhat = nlpredci(model, X, beta, ehat, J);

end


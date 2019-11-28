function [] = initialization(algorithm_name, is_no_reference, is_color, WIVC_path, LIVE_path, dependency_path)
    %% inject dependency paths
    addpath('./support_functions/');
    if (nargin == 6) && (~isempty(dependency_path))
        addpath(genpath(dependency_path));
    end

    %% compute objective scores for all images in the WIVC database
    y = compute_objective_score(algorithm_name, is_no_reference, is_color, WIVC_path);
    % non-linear mapping using the LIVE database
    sobj = compute_objective_score_live(algorithm_name, is_no_reference, is_color, LIVE_path);
    % load MOS of the LIVE database
    load('./data/live_mos.mat');
    % train non-linear maping on the LIVE database
    [beta, ehat, J] = train_nonlinear_map(sobj, live_mos);
    % run non-linear mapping on the WIVC database
    y_hat = test_nonlinear_map(y, beta, ehat, J);
    save(['./data/alg_score/' algorithm_name '.mat'], 'y_hat');

    %% generate test images
    test_result = generate_test_images(algorithm_name, y_hat, WIVC_path);
    save('./result/test_result.mat', 'test_result');
    num_pairs = size(test_result,1)-1;
    
    %% randomize playing order
    img_idx = randperm(num_pairs); %#ok
    save('./data/test_config/image_indices.mat','img_idx');
    
    %% randomize position parity
    pos_parity = randi(2,num_pairs,1) - 1; %#ok
    save('./data/test_config/position_parity.mat','pos_parity');
    
    %% remove the dependency paths
    rmpath('./support_functions/');
    if (nargin == 4) && (~isempty(dependency_path))
        rmpath(genpath(dependency_path));
    end
    disp('Initialization succeed!');
end
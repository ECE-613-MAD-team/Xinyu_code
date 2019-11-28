function pair_info = generate_test_images(algorithm_name, y_hat, WIVC_Path)
  
    in = load('./data/image_names.mat');
    sd = load('./data/seed.mat');

    rank = 6; % # of rank = 6
    file = dir('./data/scores/*.mat');
    metric_num = length(file);
    if ~exist('./data/test_images/', 'dir')
        mkdir('./data/test_images/');
    end
    
    disp('Generating test pairs...');
    % pair_info: 1. index; 2. attacker name; 3. defender name;
    % 4. quality level; 5. worst image original name; 
    % 6. best image original name; % 7. worst image saved name; 
    % 8. best image saved name; % 9. worst attacker score; 
    % 10. best attacker score.
    pair_info = cell(1, 10);
    pair_info(1,:) = {'index', 'attacker', 'defender', 'quality level', 'worst image original name', ...
        'best image original name', 'worst image saved name', 'best image saved name',...
        'worst attacker score', 'best attacker score'};
    idx = 1;
    % 1st half: current algorithm: attacker; other algorithm: defender
    for i = 1:metric_num
        attacker = algorithm_name;
        defender = file(i).name(1:end-4);
        other = load(['./data/scores/' file(i).name]);
        min_value = min(other.y_hat);
        max_value = max(other.y_hat);
        step = (max_value-min_value)/rank;
        check_points = ((min_value+(step*0.5)):step:(max_value-(step*0.5)))';
        range = [check_points-9, check_points+9];
        for j = 1:rank
            img_idx = other.y_hat>range(j,1) & other.y_hat<range(j,2);
            all_name=in.image_names(img_idx);
            all_seed=sd.seed(img_idx);
            attacker_score=y_hat(img_idx);
            [min_score,min_idx]=min(attacker_score);
            [max_score,max_idx]=max(attacker_score);
            min_name=all_name{min_idx};
            max_name=all_name{max_idx};
            min_seed=all_seed{min_idx};
            max_seed=all_seed{max_idx};
            worst_img = open_bitfield_bmp([WIVC_Path '/images/' min_name(1:end-4),'.bmp']);
            if str2double(min_name(end))~=0       
                type = str2double(min_name(end-2));
                level = str2double(min_name(end));
                worst_img = distortion_generator(worst_img, type, level, min_seed);
            end
            best_img = open_bitfield_bmp([WIVC_Path '/images/' max_name(1:end-4),'.bmp']);
            if str2double(max_name(end))~=0       
                type = str2double(max_name(end-2));
                level = str2double(max_name(end));
                best_img = distortion_generator(best_img, type, level, max_seed);
            end
            % save image for comparison and subjective experiment
            min_p_name = [attacker '_' defender '_' num2str(j) '_worst.bmp'];
            max_p_name = [attacker '_' defender '_' num2str(j) '_best.bmp'];
            imwrite(worst_img, ['./data/test_images/' min_p_name]);
            imwrite(best_img, ['./data/test_images/' max_p_name]);
            pair_info(idx+1,:) = {idx,attacker,defender,j,min_name,max_name,min_p_name,max_p_name,min_score,max_score};
            idx = idx + 1;
        end
    end
    
    % 2nd half: current algorithm: defender; other algorithm: attacker
    min_value = min(y_hat);
    max_value = max(y_hat);
    step = (max_value-min_value)/rank;
    check_points = ((min_value+(step*0.5)):step:(max_value-(step*0.5)))';
    range = [check_points-9, check_points+9];
    for i = 1:metric_num
        attacker = file(i).name(1:end-4);
        defender = algorithm_name;
        other = load(['./data/scores/' file(i).name]);
        for j = 1:rank
            img_idx = y_hat>range(j,1) & y_hat<range(j,2);
            all_name=in.image_names(img_idx);
            all_seed=sd.seed(img_idx);
            attacker_score=other.y_hat(img_idx);
            [min_score,min_idx]=min(attacker_score);
            [max_score,max_idx]=max(attacker_score);
            min_name=all_name{min_idx};
            max_name=all_name{max_idx};
            min_seed=all_seed{min_idx};
            max_seed=all_seed{max_idx};
            worst_img = open_bitfield_bmp([WIVC_Path '/images/' min_name(1:end-4),'.bmp']);
            if str2double(min_name(end))~=0       
                type = str2double(min_name(end-2));
                level = str2double(min_name(end));
                worst_img = distortion_generator(worst_img, type, level, min_seed);
            end
            best_img = open_bitfield_bmp([WIVC_Path '/images/' max_name(1:end-4),'.bmp']);
            if str2double(max_name(end))~=0       
                type = str2double(max_name(end-2));
                level = str2double(max_name(end));
                best_img = distortion_generator(best_img, type, level, max_seed);
            end
            % save image for comparison and subjective experiment
            min_p_name = [attacker '_' defender '_' num2str(j) '_worst.bmp'];
            max_p_name = [attacker '_' defender '_' num2str(j) '_best.bmp'];
            imwrite(worst_img, ['./data/test_images/' min_p_name]);
            imwrite(best_img, ['./data/test_images/' max_p_name]);
            pair_info(idx+1,:) = {idx,attacker,defender,j,min_name,max_name,min_p_name,max_p_name,min_score,max_score};
            idx = idx + 1;
        end
    end
end
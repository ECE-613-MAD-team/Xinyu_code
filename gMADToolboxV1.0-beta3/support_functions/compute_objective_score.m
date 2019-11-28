function y = compute_objective_score(algorithm_name, is_no_reference, is_color, WIVC_path)
    load('./data/seed.mat');
    file = dir([WIVC_path '/images/*.bmp']);
    if ~exist(['./data/raw_scores/' algorithm_name], 'dir')
        mkdir(['./data/raw_scores/' algorithm_name]);
    end
    y = zeros(length(file)*21,1);
    idx = 0;
    disp('Computing objective quality scores in the WIVC database (4744*21 in total). It may take a long time...');
    % no reference IQA algorithm
    if is_no_reference
        if is_color
            for i = 1:length(file)
                idx = idx + 1;
                refI = open_bitfield_bmp([WIVC_path '/images/' file(i).name]);
                y(idx) = feval(algorithm_name, refI);
                for type = 1:4
                    for level = 1:5
                        idx = idx + 1;
                        I = distortion_generator(refI, type, level, seed{idx}); %#ok
                        y(idx) = feval(algorithm_name, I);
                    end
                end
                fprintf('Finished image %d*21 / 4744*21...\n', i);
                y_segment = y(idx-20:idx); %#ok
                save(['./data/raw_scores/' algorithm_name '/'...
                    algorithm_name '_' file(i).name(1:end-4) '.mat'],'y_segment');
            end
        else
            % grayscale only
            for i = 1:length(file)
                idx = idx + 1;
                refI = open_bitfield_bmp([WIVC_path '/images/' file(i).name]);
                grayRefI = rgb2gray(refI);
                y(idx) = feval(algorithm_name, grayRefI);
                for type = 1:4
                    for level = 1:5
                        idx = idx + 1;
                        I = distortion_generator(refI, type, level, seed{idx});
                        I = rgb2gray(I);
                        y(idx) = feval(algorithm_name, I);
                    end
                end
               fprintf('Finished image %d*21 / 4744*21...\n', i);
               y_segment = y(idx-20:idx); %#ok
                save(['./data/raw_scores/' algorithm_name '/'...
                    algorithm_name '_' file(i).name(1:end-4) '.mat'],'y_segment');
            end
        end
    % full/reduced reference IQA algorithm
    % score = f(reference image, distorted image);
    else
        if is_color
            for i = 1:length(file)
                idx = idx + 1;
                refI = open_bitfield_bmp([WIVC_path '/images/' file(i).name]);
                y(idx) = feval(algorithm_name, refI, refI);
                for type = 1:4
                    for level = 1:5
                        idx = idx + 1;
                        I = distortion_generator( refI, type, level, seed{idx} );
                        y(idx) = feval(algorithm_name, refI, I);
                    end
                end
                fprintf('Finished image %d*21 / 4744*21...\n', i);
                y_segment = y(idx-20:idx); %#ok
                save(['./data/raw_scores/' algorithm_name '/'...
                    algorithm_name '_' file(i).name(1:end-4) '.mat'],'y_segment');
            end
        else
            % grayscale only
            for i = 1:length(file)
                idx = idx + 1;
                refI = open_bitfield_bmp([WIVC_path '/images/' file(i).name]);
                grayRefI = rgb2gray(refI);
                y(idx) = feval(algorithm_name, grayRefI, grayRefI);
                for type = 1:4
                    for level = 1:5
                        idx = idx + 1;
                        I = distortion_generator(refI, type, level, seed{idx});
                        I = rgb2gray(I);
                        y(idx) = feval(algorithm_name, grayRefI, I);
                    end
                end
                fprintf('Finished image %d*21 / 4744*21...\n', i);
                y_segment = y(idx-20:idx); %#ok
                save(['./data/raw_scores/' algorithm_name '/'...
                    algorithm_name '_' file(i).name(1:end-4) '.mat'],'y_segment');
            end
        end
    end
end


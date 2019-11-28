function [ img_idx ] = generate_random_imgidx( num_img, num_dup )
% This function generates random image index and randomly duplicates
% a specific number of indices, then shuffle then vector
    dups = randperm(num_img,num_dup);
    all = 1:num_img;
    img_idx = [all, dups];
    idx = randperm(length(img_idx));
    img_idx = img_idx(idx);
end


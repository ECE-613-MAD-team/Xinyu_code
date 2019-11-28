clc; clear; close all;

% 1. First, you need to put your algorithm in the root folder and set
% paths of the Waterloo Exploration database and the LIVE database.
algorithm_name = 'your algorithm name';
EXP_path = 'the path of the Waterloo Exploration database';
LIVE_path = 'the path of the LIVE database';
% if your algorithm is no-reference.
is_no_reference = 1; 
% if your algorithm takes a color image as input.
is_color = 1; 
% Note that the input range to your algorithm is 0-255 uint8 and it may
% take a really long time to compute quality scores on the Waterloo 
% Exploration database if your algorithm is not fast enough. So feel free 
% to modify the code to enable parallel computing.
initialization(algorithm_name, is_no_reference, is_color, EXP_path, LIVE_path);

% At this moment, you can find all generated image pairs under ./data/test_image,
% from which you may gain a first impression on how the models compete with
% each other.
% You can also run the subjective testing by uncommenting the following line.

% run_subjective_test();

% After finishing the subjective testing, you are able to find the data under 
% ./result/test_result.mat. 

% load('./result/test_result');

% At the very last step, go to ./support_functions/cvx and execute the 
% cvx_setup script. Now you are ready to produce a global ranking result by
% uncommenting the following line.

% data_analysis(test_result, algorithm_name);
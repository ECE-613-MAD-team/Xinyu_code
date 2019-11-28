function [ ] = run_training()
    % first set
    L1 = open_bitfield_bmp('./training/trainingL1.bmp');
    R1 = open_bitfield_bmp('./training/trainingR1.bmp');
    image_compare_gui(L1, R1, 1, 4, 'mid');

    % second set
    L2 = open_bitfield_bmp('./training/trainingL2.bmp');
    R2 = open_bitfield_bmp('./training/trainingR2.bmp');
    image_compare_gui(L2, R2, 2, 4, 'mid');

    % third set
    L3 = open_bitfield_bmp('./training/trainingL3.bmp');
    R3 = open_bitfield_bmp('./training/trainingR3.bmp');
    image_compare_gui(L3, R3, 3, 4, 'right');

    % fourth set
    L4 = open_bitfield_bmp('./training/trainingL4.bmp');
    R4 = open_bitfield_bmp('./training/trainingR4.bmp');
    image_compare_gui(L4, R4, 4, 4, 'left');

    waitfor(msgbox('Congratulations! You have completed the training sesseion!'));
end


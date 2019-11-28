function [score, status] = image_compare_gui(image1, image2, current_idx, num_pair, highlight_side)
    if(nargin < 5)
        highlight_side = 'null';
    end
    %Sets the units of your root object (screen) to pixels
    set(0,'units','pixels');
    %Obtains this pixel information
    Pix_SS = get(0,'screensize');
    screen_width = Pix_SS(3);
    screen_height = Pix_SS(4);

    score = 0;
    status = 0;
    
    % figure
    f = figure('Visible','off','Position',[0,0,screen_width,screen_height]);
    set(f,'NumberTitle','off');
    set(f,'MenuBar','none');
    set(f,'Color',[0.8,0.8,0.8]);
    
    %titleBar = sprintf('My app - Version %s', Version_Number);
    set(f, 'Name', 'GMAD');
    
    %% bottom part
    % next image pair button
    h_button_next = uicontrol('Style','pushbutton','String','Next','FontSize',12,...
          'Position',[screen_width*0.9,(1/18)*screen_height,screen_width*0.08, (1/12)*screen_height]);
    set(h_button_next,'Callback',@button_next_callback);
    
    % take a break button
    h_button_break = uicontrol('Style','pushbutton','String','Take a break','FontSize',12,...
          'Position',[screen_width*0.02 (7/72)*screen_height screen_width*0.08 (1/24)*screen_height]);
    set(h_button_break,'Callback',@button_break_callback);
    
    % slider for rating
    h_slider_scorebar = uicomponent('style','slider',-100,100,0,'position',...
        [screen_width*0.125 (1/18)*screen_height screen_width*0.75 (1/18)*screen_height],...
        'MajorTickSpacing',10,'MinorTickSpacing',5,'Paintlabels',1,'PaintTicks',1); % does not work???
    set(h_slider_scorebar, 'MouseEnteredCallback', @slider_scorebar_enter_callback);
    set(h_slider_scorebar, 'MouseWheelMovedCallback',@slider_scorebar_scroll_callback);
    set(h_slider_scorebar, 'MouseReleasedCallback', @slider_scorebar_release_callback);
    
    % edit boxes
    % should be static box with border, here is a workaround
    bgcolor_left = [1,1,1];
    bgcolor_mid = [1,1,1];
    bgcolor_right = [1,1,1];
    if(strcmp(highlight_side,'left'))
        bgcolor_left = [0,1,0];
    elseif(strcmp(highlight_side,'mid'))
        bgcolor_mid = [0,1,0];
    elseif(strcmp(highlight_side,'right'))
        bgcolor_right = [0,1,0];
    end
    
    uicontrol('Style','edit','FontSize',12,'String','Left is better','enable','inactive',...
        'BackgroundColor',bgcolor_left,...
        'Position',[0.125*screen_width 1/9*screen_height 0.3*screen_width 1/36*screen_height]);
    
    uicontrol('Style','edit','FontSize',12,'String','Uncentain','enable','inactive',...
        'BackgroundColor',bgcolor_mid,...
        'Position',[0.425*screen_width 1/9*screen_height 0.15*screen_width 1/36*screen_height]);
    
    uicontrol('Style','edit','FontSize',12,'String','Right is better','enable','inactive',...
        'BackgroundColor',bgcolor_right,...
        'Position',[0.575*screen_width 1/9*screen_height 0.3*screen_width 1/36*screen_height]);
    
    %uicontrol('Style','text','BackgroundColor',[0.925,0.925,0.925],...
    %    'Position',[0.10*screen_width (1/18)*screen_height 0.025*screen_width (1/12)*screen_height]);
    
    %uicontrol('Style','text','BackgroundColor',[0.925,0.925 0.925],...
    %    'Position',[0.875*screen_width (1/18)*screen_height 0.025*screen_width (1/12)*screen_height]);
    
    [height1, width1, ~] = size(image1);
    [height2, width2, ~] = size(image2);
    
    l1 = 0.25*screen_width - 0.5*width1;
    l1 = l1/screen_width;
    b1 = (101/180)*screen_height - 0.5*height1;
    b1 = b1/screen_height;
    l2 = 0.75*screen_width - 0.5*width2;
    l2 = l2/screen_width;
    b2 = (101/180)*screen_height - 0.5*height2;
    b2 = b2/screen_height;
    
    % Create a progress bar
    jProgressBar = javax.swing.JProgressBar;
    set(jProgressBar, 'StringPainted', 1);
    set(jProgressBar, 'Value', (current_idx/num_pair)*100);
    javacomponent(jProgressBar, [screen_width*0.02 (1/18)*screen_height screen_width*0.08 (1/24)*screen_height]);
    
    %% top part
    subplot(1,2,1,'Position',[l1 b1 width1/screen_width height1/screen_height]);
    imshow(image1);
    subplot(1,2,2,'Position',[l2 b2 width2/screen_width height2/screen_height]);
    imshow(image2);
    
    set(f,'Visible','on');
    
    done = 0;
    while done == 0
        pause(.02);                    % <-- This pause gives the callback
    end    
    
    %% callback functions
    function button_next_callback(~,~) 
        done = 1;
    end

    function button_break_callback(~,~)
        status = 1;
        done = 1;
    end

    function slider_scorebar_enter_callback(source,~)
        source.requestFocus();
    end
    
    function slider_scorebar_scroll_callback(source,eventdata)
        val = get(eventdata,'wheelRotation');
        score = get(source, 'Value');
        set(source, 'Value', score + val);
        score = get(source, 'Value');
    end

    function slider_scorebar_release_callback(source,~) 
        score = get(source, 'Value');
    end

    close(f);
end
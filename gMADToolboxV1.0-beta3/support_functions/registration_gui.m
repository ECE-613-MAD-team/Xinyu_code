function [status, subject_profile] = registration_gui
    status = 0;
    firstname ='';
    lastname = '';
    email ='';
    gender = '';
    year = 2015;
    month = 1;
    date = 1;
    subject_profile = cell(1,7);
    
    nda_msg = ['Any records that contain your identity will be treated as confidential, ' ...
    'accessed only by the researchers. ' 'The record will not be used for a commercial purpose.'];
    
    % figure
    f = figure('Visible','off','Units','Normalized','Position',[0,0,0.5,0.5]);
    movegui(f,'center')
    set(f,'NumberTitle','off');
    set(f,'MenuBar','none');
    set(f,'Name','Registration');
    set(f,'Color',[1,1,1]);
    
    % buttons
    h_button_confirm = uicontrol('Style','pushbutton','String','Confirm','FontSize',12,...
        'Units','Normalized','Position',[0.2, 0.1, 0.2, 0.1]);
    set(h_button_confirm,'Callback',@button_confirm_callback);
    
    h_button_cancel = uicontrol('Style','pushbutton','String','Cancel','FontSize',12,...
        'Units','Normalized','Position',[0.6, 0.1, 0.2, 0.1]);
    set(h_button_cancel,'Callback',@button_cancel_callback);
    
    % birthday panel
    h_panel_birthday = uipanel('Title','Birthday','FontSize',12,...
        'BackgroundColor',[1,1,1],'Position',[0.1 0.45 0.8 0.2]);
    
    % static texts
    uicontrol('Style','text','String','*','FontSize',12,'BackgroundColor',[1,1,1],...
        'ForegroundColor',[1,0,0],'Units','Normalized','Position',[0.08, 0.815, 0.01, 0.1]);
    
    uicontrol('Style','text','String','First name:','FontSize',12,'BackgroundColor',[1,1,1],...
        'Units','Normalized','Position',[0.09, 0.82, 0.1, 0.1]);
    
    uicontrol('Style','text','String','*','FontSize',12,'BackgroundColor',[1,1,1],...
        'ForegroundColor',[1,0,0],'Units','Normalized','Position',[0.08, 0.665, 0.01, 0.1]);
    
    uicontrol('Style','text','String','Last name:','FontSize',12,'BackgroundColor',[1,1,1],...
        'Units','Normalized','Position',[0.09, 0.67, 0.1, 0.1]);
    
    uicontrol('Style','text','String','E-mail:','FontSize',12,'BackgroundColor',[1,1,1],...
        'Units','Normalized','Position',[0.5, 0.82, 0.1, 0.1]);
    
    uicontrol('Style','text','String','*','FontSize',12,'BackgroundColor',[1,1,1],...
        'ForegroundColor',[1,0,0],'Units','Normalized','Position',[0.49, 0.665, 0.01, 0.1]);
    
    uicontrol('Style','text','String','Gender:','FontSize',12,'BackgroundColor',[1,1,1],...
        'Units','Normalized','Position',[0.5, 0.67, 0.1, 0.1]);
    
    uicontrol('Style','text','String',nda_msg,'FontSize',12,'BackgroundColor',[1,1,1],...
        'FontWeight','bold','Units','Normalized','Position',[0.1, 0.25, 0.8, 0.1]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','*','FontSize',12,...
        'BackgroundColor',[1,1,1],'ForegroundColor',[1,0,0],'Units','Normalized',...
        'Position',[0.09, 0.05, 0.01, 0.5]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','Year','FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.1, 0.1, 0.1, 0.5]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','*','FontSize',12,...
        'BackgroundColor',[1,1,1],'ForegroundColor',[1,0,0],'Units','Normalized',...
        'Position',[0.36, 0.05, 0.01, 0.5]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','Month','FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.37, 0.1, 0.1, 0.5]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','*','FontSize',12,...
        'BackgroundColor',[1,1,1],'ForegroundColor',[1,0,0],'Units','Normalized',...
        'Position',[0.64, 0.05, 0.01, 0.5]);
    
    uicontrol('Parent',h_panel_birthday,'Style','text','String','Date','FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.65, 0.1, 0.1, 0.5]);
    
    % edit boxes
    uicontrol('Style','edit','Callback',@edit_firstname_callback,'FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.2, 0.85, 0.2, 0.1]);
    
    uicontrol('Style','edit','Callback',@edit_lastname_callback,'FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.2, 0.7, 0.2, 0.1]);
    
    uicontrol('Style','edit','Callback',@edit_email_callback,'FontSize',12,...
        'BackgroundColor',[1,1,1],'Units','Normalized','Position',[0.6, 0.85, 0.25, 0.1]);
    
    h_button_group = uibuttongroup('Units','Normalized','BackgroundColor',[1,1,1],'Position',[0.6 0.7 0.25 0.1]);
    
    uicontrol('Style','Radio','Parent',h_button_group,'HandleVisibility','off','String','Male', ...
        'BackgroundColor',[1,1,1],'FontSize',12,'Units','Normalized','Position',[0.1 0 0.5 1]);
    
    uicontrol('Style','Radio','Parent',h_button_group,'HandleVisibility','off','String','Female', ...
        'BackgroundColor',[1,1,1],'FontSize',12,'Units','Normalized','Position',[0.6 0 0.4 1]);
    
    set(h_button_group,'SelectionChangeFcn',@select_gender_callback);
	set(h_button_group,'SelectedObject',[]);  % No selection
    
    % list boxes
    years = 2015:-1:1930;
    uicontrol('Parent',h_panel_birthday,'Style','popup','String',cellstr(num2str(years'))',...
        'BackgroundColor',[1,1,1],'Callback', @popup_year_callback,...
        'FontSize',12,'Units','Normalized','Position',[0.2, 0.55, 0.1, 0.1]);
       
    months = 1:12;
    uicontrol('Parent',h_panel_birthday,'Style','popup','String',cellstr(num2str(months'))',...
        'BackgroundColor',[1,1,1],'Callback', @popup_month_callback,...
        'FontSize',12,'Units','Normalized','Position',[0.475, 0.55, 0.1, 0.1]);
       
    dates = 1:31;
    uicontrol('Parent',h_panel_birthday,'Style','popup','String',cellstr(num2str(dates'))',...
        'BackgroundColor',[1,1,1],'Callback',@popup_date_callback,...
        'FontSize',12,'Units','Normalized','Position',[0.75, 0.55, 0.1, 0.1]);
    
    set(f,'Visible','on');
    
    while status == 0
        pause(.02);                    % <-- This pause gives the callback
    end    
    
    %% callback functions
    function button_confirm_callback(~,~)
        % completeness check
        if (strcmp(firstname,''))
            msgbox('Please enter your first name!','Error','error');
            return;
        end
        
        if (strcmp(lastname,''))
            msgbox('Please enter your last name!','Error','error');
            return;
        end
        
        if (strcmp(gender,''))
            msgbox('Please select your gender!','Error','error');
            return;
        end
        
        subject_profile{1} = firstname;
        subject_profile{2} = lastname;
        subject_profile{3} = email;
        subject_profile{4} = gender;
        subject_profile{5} = year;
        subject_profile{6} = month;
        subject_profile{7} = date;
        status = 1;
    end

    function button_cancel_callback(~,~)
        status = -1;
    end

    function edit_firstname_callback(source,~)
        firstname = get(source,'String');
    end

    function edit_lastname_callback(source,~)
        lastname = get(source,'String');
    end

    function edit_email_callback(source,~)
        email = get(source,'String');
    end

    function select_gender_callback(source,~)
        gender = get(get(source,'SelectedObject'),'String');
    end

    function popup_year_callback(source,~)
        str = get(source,'String');
        val = get(source,'Value');
        year = str2double(str{val});
    end

    function popup_month_callback(source,~)
        str = get(source,'String');
        val = get(source,'Value');
        month = str2double(str{val});
    end

    function popup_date_callback(source,~)
        str = get(source,'String');
        val = get(source,'Value');
        date = str2double(str{val});
    end

    close(f);
end
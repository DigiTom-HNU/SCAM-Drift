function [C_data,S_data,colNum] = candsInfoRead(path)
   % path = 'C:\Users\shmt\Desktop\连续扫描筛选有效帧小程序';
    displaceFile = dir(fullfile(path, '*info.txt'));
    if isempty(displaceFile)
            error('无位移台信息文件！');
    end


    C_filename = displaceFile(1,1).name;                                       % 相机扫描时间与帧数对应信息txt文件名
    S_filename = displaceFile(2,1).name;                                       % 位移台扫描时间与位置对应信息txt文件名
    C_folder = displaceFile(1,1).folder;                                       % 相机扫描时间与帧数对应信息txt路径
    S_folder = displaceFile(2,1).folder;                                       % 位移台扫描时间与位置对应信息txt路径

    % 先把位移台的信息提取出来
    S_fileID = fopen(fullfile(S_folder, S_filename),'r');                      % 打开txt文件
    S_data_o = textscan(S_fileID, '%s %s %s %s %s');         % 将数据存到S_data_o中，S_data_o是元胞
    S_cutline_loc = find(ismember(S_data_o{1},'----------------------------------------------------------'));       %分割线在第几行，分割线上面数两行的启示XYZ坐标，分割线后一行是移动过程中X的坐标
    colNum = length(S_cutline_loc);                                            % 总共有几列
    S_data_length = length(S_data_o{1}) - colNum * 2;                          % 所有列中有效数据一共有多少个,减去3是因为要刨掉分割线、空的一行和XYZ起始位置的一行
    S_data = zeros(S_data_length,colNum);                                      % 建立一个空矩阵，col1是列数，col2是时间col345是XYZ
    for i = 1:colNum                                                           %每一列的数据长度
        if i < colNum
            S_r(i) = S_cutline_loc(i+1) - S_cutline_loc(i) - 2;
        elseif i == colNum
            S_r(i)  = length(S_data_o{1}) - S_cutline_loc(i);
        end
    end
    %  创建几个空向量，把列数、时间、XYZ信息分别提取出来赋值到S_data
    v1 = [];
    v2 = [];
    v3 = [];
    v4 = [];
    v5 = [];
    for i = 1:colNum
        v1 = [v1;ones(S_r(i),1)*i];
        v2 = [v2;str2double(convertCharsToStrings(S_data_o{1}(S_cutline_loc(i)+1:S_cutline_loc(i)+S_r(i))))];
        v3 = [v3;str2double(convertCharsToStrings(S_data_o{2}(S_cutline_loc(i)+1:S_cutline_loc(i)+S_r(i))))];
        v4 = [v4;ones(S_r(i),1).*(str2double(convertCharsToStrings(S_data_o{2}(S_cutline_loc(i)-1))))];
        v5 = [v5;ones(S_r(i),1).*(str2double(convertCharsToStrings(S_data_o{3}(S_cutline_loc(i)-1))))];
    end
    S_data(:,1) = v1;                                                          % 表示扫描的第几列
    S_data(:,2) = v2;                                                          % 表示对应的计算机时间，单位是ms
    S_data(:,3) = v3;                                                          % 表示位移台X轴信息
    S_data(:,4) = v4;                                                          % 表示位移台Y轴信息
    S_data(:,5) = v5;                                                          % 表示位移台Z轴信息
    clear v1 v2 v3 v4 v5                                                       % 清空中间变量

    % 再把相机的信息提取出来
    C_fileID = fopen(fullfile(C_folder, C_filename),'r');                      % 打开txt文件
    C_data_o = textscan(C_fileID, '%s %s', 'Delimiter', ' ');                  % 将数据存到C_data_o中，C_data_o是元胞
    C_cutline_loc = find(ismember(C_data_o{1},'----------------------------------------------------------'));       % v1变量1是找到位移台txt文件中的分割线在第几行，分割线上面是这一列的其实XYZ坐标，分割线后是移动过程中X的坐标
    %  创建几个空向量，把列数、时间、XYZ信息分别提取出来赋值到C_data
    C_data_length = length(C_data_o{1}) - colNum;                              % 所有列中有效数据一共有多少个
    C_data = zeros(C_data_length,3);                                           % 建立一个空矩阵，col1是列数，col2是时间col3帧数
    for i = 1:colNum                                                           %每一列的数据长度
        if i < colNum
            C_r(i) = C_cutline_loc(i+1) - C_cutline_loc(i) - 1;
        elseif i == colNum
            C_r(i)  = length(C_data_o{1}) - C_cutline_loc(i);
        end
    end
    %  创建几个空向量，把列数、时间、帧数序号分别提取出来赋值到C_data
    v1 = [];
    v2 = [];
    v3 = [];
    for i = 1:colNum
        v1 = [v1;ones(C_r(i),1)*i];
        v2 = [v2;str2double(convertCharsToStrings((C_data_o{2}(C_cutline_loc(i)+1:C_cutline_loc(i)+C_r(i)))))];
        v3 = [v3;(1:1:C_r(i))'];
    end
    C_data(:,1) = v1;                                                          % 表示扫描的第几列
    C_data(:,2) = v2;                                                          % 表示对应的计算机时间，单位是ms
    C_data(:,3) = v3;                                                          % 表示帧数

    

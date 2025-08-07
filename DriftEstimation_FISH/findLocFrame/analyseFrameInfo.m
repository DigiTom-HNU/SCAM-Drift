function [frameInfos] = analyseFrameInfo(data_eff,path,endFrames)
%     path = 'J:\1';
%     if ~exist(fullfile(path,'locFile'),'dir')
%         mkdir(fullfile(path,'locFile'));                                   % 创建这个路径
%         fprintf('文件路径 %s 创建成功！\n',fullfile(path,'locFile'))
%     else
%         fprintf('文件路径 %s 已存在！\n',fullfile(path,'locFile'))
%     end
     
    fileInfo = dir(fullfile(path, '\locFile1','*continuousScan*.mat'));                % 看一下这个文件夹里有多少个stack文件
    frameInfos = [];% 创建一个元胞用来存表格
    fileName = strings(length(fileInfo),1);
    colId = zeros(length(fileInfo),1);
    rowId = zeros(length(fileInfo),1);
    for i = 1:length(fileInfo)
        fileName(i)  = string(fileInfo(i).name);
        startColId = 'continuousScan_(\d+)_part';
        colId(i)  = str2double(regexp(fileInfo(i).name, startColId, 'tokens', 'once'));
        startRowId = strfind(fileInfo(i).name,'.mat');
        rowId(i)  = str2double(fileInfo(i).name(startRowId-5:startRowId-1));
    end

    for ii = min(colId):max(colId)                                                      % 遍历这些列
        [x1,~] = find(data_eff(:,1) == ii);                               % 找每列里面有几个part
        stackpartStartNum = min(rowId(colId == ii)); 
        stackpartEndNum = max(rowId(colId == ii));                           % 每列有几个part
        col_starttime = min(min(x1));                                      % part的起始编号
        col_endtime = max(max(x1));                                        % part的结束编号
 
        for jj = stackpartStartNum:stackpartEndNum                                            % 遍历每个part
            [x2,~] = find(data_eff(col_starttime:col_endtime,4) == jj);   
            stack_starttime = min(min(x2)) + col_starttime - 1;            % 找到每个part第一帧
            stack_endtime = max(max(x2)) + col_starttime - 1;              % 找到每个part最后一帧
%             if (ii-1)*10+jj+1>length(fileInfo)
%                 continue;
%             end            
            filenameTemp = cellstr(fileName(colId == ii & rowId == jj));       % frameInfo第一列存文件名
            pattern = 'continuousScan_(\d+)';
            matches = regexp(filenameTemp{1}, pattern, 'tokens');
            StackCol = str2double(matches{1});
            if jj<stackpartEndNum                                             % 判断是不是最后一个part
                frames = 1024;       % 如果不是的话每个part里面有512帧原始数据
            else
                frames = endFrames;       % 如果是的话，最后一个part有413帧原始数据
            end
            effectframes = stack_endtime-stack_starttime+1;       % 第三列是该part中有多少有效的帧
            effectstartframe = data_eff(stack_starttime,5);           % 第四列是该part有效帧起始的帧数
            effectendframe = data_eff(stack_endtime,5);             % 第五列是该part有效帧结束的帧数
            frameInfos = [frameInfos;[StackCol,frames,effectframes,effectstartframe,effectendframe]];
        end
        
    end

%     writecell(frameInfo,fullfile(path,'frameInfo.txt'),'Delimiter','\t');
%     type frameInfo.txt;
        
function [data_eff,frameInfo] = CameraAndStageInfo(path)
stackFrameNum = 1024;
endFrames = 1024;

% 读取文件
%     [C_data,S_data,colNum] = candsInfoRead1(path);
[C_data,S_data,colNum] = candsInfoRead(path);
% 筛选有用的帧数，输出矩阵第一列为扫描的列数，第二列为对应的计算机时间，单位为毫秒，第三列为这一列中的第几帧，第四行为在该列stark中的paro几，第五列在该stack中的第几帧，六七八分别为XYZ坐标

% 加速之后
data_eff = simplificationCuda(C_data,S_data,colNum,stackFrameNum);
data_eff = data_eff(:,1:8);
frameInfo = analyseFrameInfo(data_eff,path,endFrames);
    



% ColId = 'continuousScan_(\d+)_part';
% RowId = 'part_(\d+)';
% colid = str2double(regexp(filename, ColId, 'tokens', 'once'));
% rowid = str2double(regexp(filename, RowId, 'tokens', 'once'));
% data_eff = data_eff(data_eff(:,1) == colid & data_eff(:,4) == rowid,:);

    
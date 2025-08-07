function [data_eff] = simplificationCuda(C_data,S_data,colNum,stackFrameNum)
    for ii = 1:colNum                                                      % 一列一列的筛选
        [colnumber,index] = find(S_data(:,1)==ii);                         % 判断该列有多少坐标数据
        startLoc = min(min(colnumber));                                    % 找到该列起始的坐标数据
        endLoc = max(max(colnumber));                                      % 找到该列结束的坐标数据
        % 使用cuda进行处理
        outputSdata_t = [];
        inputSdata = S_data(startLoc:endLoc,:);
        outputSdata = filtrateSdataCuda(inputSdata);
        outputSdata(outputSdata(:,1)==0,:)=[];
        outputSdata = correcttimeIntevalCuda(outputSdata);
%         [idx,idy]=find(outputSdata(:,6)==max(outputSdata(:,6)));
%         outputSdata(end,6)=1;
        total_num = sum(outputSdata(:,6));
        outputSdata_t=fillupSdataCuda(outputSdata,total_num);
        outputSdata_t(1,:)=outputSdata(1,1:colNum);
        if ii == 1                                                         % 如果是第一列的话
            S_data_eff = outputSdata_t;                                               % 那位移台的有效数据就等于v1
        elseif ii>1                                                        % 如果不是第一列的话
            S_data_eff = [S_data_eff;outputSdata_t];                                  % 那就在后面并入最新的v1
        end
       
        [mintime,~] = find(outputSdata_t(:,2) == min(min(outputSdata_t(:,2))));                  % 找到这个v1中最小的时间点的位置
        [maxtime,~] = find(outputSdata_t(:,2) == max(max(outputSdata_t(:,2))));                  % 找到这个v1中最大的时间点的位置
        mintime = outputSdata_t(mintime,2);                                           % 记录下这一列开始移动的时间
        maxtime = outputSdata_t(maxtime,2);                                           % 记录下这一列停止移动的时间
        C_dataNum = sum(C_data(:,1)==ii);                                  % 统计一下这一列移动了多少毫秒
        inputCdata = C_data((C_dataNum*(ii-1)+1):(C_dataNum*ii),:);
        outputCdata = filtrataCdataCuda(inputCdata, mintime, maxtime);
        outputCdata(outputCdata(:,1)==0,:)=[];
        if ii == 1                                                         % 如果是第一列的话
            C_data_eff = outputCdata;                                               % 那相机的有效数据就等于v2
        elseif ii>1                                                        % 如果不是第一列的话
            C_data_eff = [C_data_eff;outputCdata];                                  % 那就在后面并入最新的v2
        end
    end
%     singlestacknum=stackFrameNum;
    % 将最终的数据合并到data_eff里面，第一列为扫描列数，第二列为时间，第三列为帧数，第四-六列分别为XYZ坐标，
    data_eff=simplificationInfoCuda(C_data_eff,S_data_eff,stackFrameNum);    
    data_eff=data_eff(2:end-1,:);

    
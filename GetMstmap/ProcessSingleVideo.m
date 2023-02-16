function MSTmap_whole_video = ProcessSingleVideo(video_dir)
    %% 获取视频
    video_file = strcat(video_dir, "/video.avi");
    obj = VideoReader(video_file);
    numFrames = obj.NumberOfFrames;

    MSTmap_whole_video = zeros(63, numFrames, 6);  % 63 x T x 6
    landmark_num = 81;

    %% 逐帧处理
    cnt = 0;  % 记录检测不到特征点的帧数
    for k = 1 : numFrames
        frame = read(obj,k);
        % load landmarks
        lmk_path = strcat(video_dir, '/landmarks/', num2str(k - 1), '.txt');  % k - 1 !
        lmks = int32(load(lmk_path));
        % 未检测到特征点, 按 0 处理
        if lmks(1, 1) == -1
            cnt = cnt + 1;
            continue
        end
        MSTmap_whole_video = GenerateSignalMap(MSTmap_whole_video, frame, k, lmks, landmark_num);
    end
    if cnt > 0
        msg = sprintf('cnt = %d\n', cnt);
        fprintf(msg);
    end
end
%% 获取所有视频路径
data_paths = ["/share1/home/liangqian/database/VIPL-HR/data"];
subject_paths = readPath(data_paths);
task_paths = readPath(subject_paths);
video_paths = readPath(task_paths);
clip_length = 300;  % !
%% 遍历视频, 逐个处理
tic;
for i = 1:length(video_paths)
    % calculate
    mstmap = ProcessSingleVideo(video_paths(i, 1));  % 63 x T x 6
    % load other information
    gt_path = strcat(video_paths(i, 1), "/gt_HR.csv");
    gt = readtable(gt_path);
    gt = gt.Variables;

    time_path = strcat(video_paths(i, 1), "/time.txt");
    if exist(time_path, "file")
        time = load(time_path);
        fps = length(time) / (time(end) - time(1)) * 1000;
    else
        fps = 30.0;
    end

    bvp_path = strcat(video_paths(i, 1), "/wave.csv");
    bvp = readtable(bvp_path);
    bvp = bvp.Variables;

    %% 保存原始样本
    % save
    % save_path = strcat(video_paths(i, 1), "/cvd_cache");
    % mkdir(save_path);
    % save_MSTmaps(save_path, mstmap, bvp, gt, fps, clip_length);

    %% 保存上采样样本
    save_path = strcat(video_paths(i, 1), "/up_samples");
    save_up(save_path, mstmap, bvp, gt, fps, 200);  % 200 -> 300
    %% 保存下采样样本
    save_path = strcat(video_paths(i, 1), "/down_samples");
    save_down(save_path, mstmap, bvp, gt, fps, 450);  % 450 -> 300

    % print progress
    cost = toc;  % 已使用的时间 (s)
    remain = cost / i * (length(video_paths) - i) / 3600;  % 预计剩余时间 (h)
    msg = sprintf('%d / %d done, %.3f hours remained\n', i, length(video_paths), remain);
    fprintf(msg);
end
exit();  % 防止 nohup 输出错误

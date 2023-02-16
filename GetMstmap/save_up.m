function save_up(HR_train_path, SignalMap, BVP_all, gt, fps, clip_length)
    %% 采样函数, 由 clip_length 采样至 300, 需要插值 mstmap 和 bvp, bpm 也要改变
    img_num = size(SignalMap,2);
    channel_num = size(SignalMap,3);

    clip_num = floor((img_num - clip_length)/fps*2);

    for i = 1:clip_num
        begin_idx = floor(0.5*fps*(i-1)+1);
        
        if (begin_idx + clip_length - 1 > img_num)
            continue;
        end
        
        if floor(begin_idx/fps) >= length(gt)
            continue;
        end
        
        gt_temp = mean(gt(max(1,floor(begin_idx/fps)):min(length(gt), floor((begin_idx+clip_length)/fps))));
        
        % 上采样 clip 会导致 hr 下降, 排除合理区间外的 clip
        if gt_temp > 85 || gt_temp < 70
            continue;
        end

        final_signal = SignalMap(:, begin_idx: begin_idx + clip_length - 1, :);
        judge = mean(final_signal,1);
        if ~isempty(find(judge(1,:,2) == 0))
            continue;
        else
            save_path = strcat(HR_train_path, '/', num2str(i));
            if ~exist(save_path)
                mkdir(save_path);
            end

            bpm = gt_temp * clip_length / fps / 60;
            bvp_begin = floor(begin_idx / img_num * length(BVP_all));
            bvp_len = round(clip_length / img_num * length(BVP_all));
            if(bvp_begin + bvp_len > length(BVP_all))
                continue;
            end

            % bpm 变为原来的 2/3
            bpm = bpm * 2 / 3;
            % 根据采样率进行插值
            bvp = BVP_all(bvp_begin:bvp_begin+bvp_len);
            x = 1:length(bvp);
            xx = 1:300;  % 目标长度
            xx = xx*length(bvp)/300;
            bvp = interp1(x, bvp, xx);

            label_path = strcat(save_path, '/gt.mat');
            fps_path = strcat(save_path, '/fps.mat');
            bpm_path = strcat(save_path, '/bpm.mat');
            bvp_path = strcat(save_path, '/bvp.mat');

            % eval(['save ', label_path, ' gt_temp']);
            % eval(['save ', fps_path, ' fps']);
            % eval(['save ', bpm_path, ' bpm']);
            % eval(['save ', bvp_path, ' bvp']);
            save(label_path, "gt_temp");
            save(fps_path, "fps");
            save(bpm_path, "bpm");
            save(bvp_path, "bvp");

            final_signal1 = final_signal;
            for idx = 1:size(final_signal,1)
                for c = 1:channel_num
                    temp = final_signal(idx,:,c);
                    temp = movmean(temp,3);
                    final_signal1(idx,:,c) = (temp - min(temp))/(max(temp) - min(temp))*255;
                end
            end
            
            % 将 MSTmap 上采样
            final_signal1 = imresize(final_signal1, [63, 300]);

            img1 = final_signal1(:,:,[1 2 3]);
            img2 = final_signal1(:,:,[4 5 6]);
            img_mat = final_signal;

            % img1_path = strcat(save_path, '/img_rgb.png');
            % img2_path = strcat(save_path, '/img_yuv.png');
            % imwrite(uint8(img1), img1_path);
            % imwrite(uint8(img2), img2_path);

            img1_path = strcat(save_path, '/img_rgb.mat');
            img2_path = strcat(save_path, '/img_yuv.mat');
            img1 = uint8(img1);
            img2 = uint8(img2);
            save(img1_path, "img1");
            save(img2_path, "img2");
        end
    end

end
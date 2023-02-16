function SignalMap = GenerateSignalMap(SignalMap, img, idx, lmks, lmk_num)

    [m,n,c] = size(img);

    %% R G B Y U V
    R = img(:,:,1);
    G = img(:,:,2);
    B = img(:,:,3);
    
    % Conversion Formula
    Y = 0.299 * R + 0.587 * G + 0.114 * B;
    U =128 - 0.168736 * R - 0.331264 * G + 0.5 * B;
    V =128 + 0.5 * R - 0.418688 * G - 0.081312 * B;
    
    img(:,:,4) = Y;
    img(:,:,5) = U;
    img(:,:,6) = V;
    
    %% get ROIs 
    if lmk_num == 81
        ROI_cheek_left1 = [65 66 67 38 36 65] + 1;
        ROI_cheek_left2 = [67 68 69 70 46 40 38 67] + 1;
        ROI_cheek_right1 = [73 74 75 39 37 73] + 1;
        ROI_cheek_right2 = [75 76 77 78 47 41 39 75] + 1;
        ROI_mouth = [70 71 72 64 80 79 78 47 59 55 58 46 70] + 1;
        ROI_forehead = [18 22 20 24 19 26 30 28 32 27] + 1;
        
        forehead = lmks(ROI_forehead, :);
        % forehead = lmks(:,ROI_forehead);
        eye_distance = norm(double(lmks(1, :) - lmks(10, :)));
        % eye_distance = norm(lmks(:,1) - lmks(:,10));
        tmp = (mean(lmks(19:26, :))+ mean(lmks(27:34, :)))/2 - (mean(lmks(1:9, :))+mean(lmks(10:18, :)))/2;
        % tmp = (mean(lmks(:,19:26)')+ mean(lmks(:,27:34)'))/2 - (mean(lmks(:,1:9)')+mean(lmks(:,10:18)'))/2;
        tmp = int32(eye_distance/norm(tmp)*0.6*tmp);
        % tmp = eye_distance/norm(tmp)*0.6*tmp';
        ROI_forehead = [forehead; forehead(end, :)+tmp; forehead(1, :)+tmp; forehead(1, :)];
        % ROI_forehead = [forehead forehead(:,end)+tmp forehead(:,1)+tmp forehead(:,1)];
    end
    
    % mask_ROI_cheek_left1 = poly2mask(lmks(1,ROI_cheek_left1),lmks(2,ROI_cheek_left1),size(img,1), size(img,2));
    % mask_ROI_cheek_left2 = poly2mask(lmks(1,ROI_cheek_left2),lmks(2,ROI_cheek_left2),size(img,1), size(img,2));
    % mask_ROI_cheek_right1 = poly2mask(lmks(1,ROI_cheek_right1),lmks(2,ROI_cheek_right1),size(img,1), size(img,2));
    % mask_ROI_cheek_right2 = poly2mask(lmks(1,ROI_cheek_right2),lmks(2,ROI_cheek_right2),size(img,1), size(img,2));
    % mask_ROI_mouth = poly2mask(lmks(1,ROI_mouth),lmks(2,ROI_mouth),size(img,1), size(img,2));
    % mask_ROI_forehead = poly2mask(ROI_forehead(1,:),ROI_forehead(2,:),size(img,1), size(img,2));

    mask_ROI_cheek_left1 = poly2mask(double(lmks(ROI_cheek_left1, 1)), double(lmks(ROI_cheek_left1, 2)), size(img, 1), size(img, 2));
    mask_ROI_cheek_left2 = poly2mask(double(lmks(ROI_cheek_left2, 1)), double(lmks(ROI_cheek_left2, 2)), size(img, 1), size(img, 2));
    mask_ROI_cheek_right1 = poly2mask(double(lmks(ROI_cheek_right1, 1)), double(lmks(ROI_cheek_right1, 2)), size(img, 1), size(img, 2));
    mask_ROI_cheek_right2 = poly2mask(double(lmks(ROI_cheek_right2, 1)), double(lmks(ROI_cheek_right2, 2)), size(img, 1), size(img, 2));
    mask_ROI_mouth = poly2mask(double(lmks(ROI_mouth, 1)), double(lmks(ROI_mouth, 2)), size(img, 1), size(img, 2));
    mask_ROI_forehead = poly2mask(double(ROI_forehead(:, 1)), double(ROI_forehead(:, 2)), size(img, 1), size(img, 2));
    
    % 可视化, 检查结果
    % figure(1);
    % clf;
    % imshow(img(:,:,1:3));
    % hold on;
    % plot(lmks(ROI_cheek_left1, 1),lmks(ROI_cheek_left1, 2), 'Linewidth',2)
    % plot(lmks(ROI_cheek_left2, 1),lmks(ROI_cheek_left2, 2), 'Linewidth',2)
    % plot(lmks(ROI_cheek_right1, 1),lmks(ROI_cheek_right1, 2), 'Linewidth',2)
    % plot(lmks(ROI_cheek_right2, 1),lmks(ROI_cheek_right2, 2), 'Linewidth',2)
    % plot(lmks(ROI_mouth, 1),lmks(ROI_mouth, 2), 'Linewidth',2)
    % plot(ROI_forehead(:, 1), ROI_forehead(:, 2), 'Linewidth',2)
    % saveas(gcf,'result.png')

    Signal_tmp = zeros(6, 6);
    ROI_num = zeros(6, 1);
%% 按 ROI 计算
    Signal_tmp(1,:) = get_ROI_signal(img, mask_ROI_cheek_left1);
    Signal_tmp(2,:) = get_ROI_signal(img, mask_ROI_cheek_left2);
    Signal_tmp(3,:) = get_ROI_signal(img, mask_ROI_cheek_right1);
    Signal_tmp(4,:) = get_ROI_signal(img, mask_ROI_cheek_right2);
    Signal_tmp(5,:) = get_ROI_signal(img, mask_ROI_mouth);
    Signal_tmp(6,:) = get_ROI_signal(img, mask_ROI_forehead);
    
%% get ROI pixel
    ROI_num(1) = length(find(mask_ROI_cheek_left1 == 1));
    ROI_num(2) = length(find(mask_ROI_cheek_left2 == 1));
    ROI_num(3) = length(find(mask_ROI_cheek_right1 == 1));
    ROI_num(4) = length(find(mask_ROI_cheek_right2 == 1));
    ROI_num(5) = length(find(mask_ROI_mouth == 1));
    ROI_num(6) = length(find(mask_ROI_forehead == 1));
    
%% 多层次结合 ROI
    SignalMap(:,idx,:) = getCombinedSignalMap(Signal_tmp, ROI_num);
end
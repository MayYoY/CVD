function sub_paths = readPath(base_paths)
    sub_paths = [];
    for i = 1:length(base_paths)  % read base paths one by one
        base_path = base_paths(i, 1);
        files = dir(base_path);
        for j = 1:length(files)
            % ignore . and .. file
            if strcmp(files(j).name,'.') == 1 || strcmp(files(j).name,'..') == 1
                continue
            else
                sub_path = strcat(base_path, "/", files(j).name);
                sub_paths = [sub_paths; sub_path];
            end
        end
    end
end
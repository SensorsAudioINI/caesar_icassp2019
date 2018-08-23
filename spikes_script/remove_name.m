function new_path = remove_name(path)
    
    a = strsplit(path, '\');
    b = strjoin(a(1:end-1), '\');
    
    new_path = [b '\'];
end
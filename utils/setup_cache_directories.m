function [p] = setup_cache_directories(p)

if (exist(p.results_dir,'dir'))
    user_choice = input('A previous results directory corresponding to the same experiment codename exists, should I go on and delete it? (yes/no) ','s');
    if(~strcmpi(user_choice,'yes'))
        suffix = input('Please enter a new suffix for the results directory: ','s');
        p.codename = [p.codename,'_',suffix];
        p.results_dir = [p.results_dir,'_',suffix];
    else
        fprintf('REMOVING previously existing results path\n');
        [status,message,messageid] = rmdir(p.results_dir,'s'); %#ok<*NASGU,*ASGLU>
    end
end

if(~isempty(p.results_dir))
    
    [status,message,messageid] = mkdir(p.results_dir);
    
    p.test_subdir_path = fullfile(p.results_dir,'final_results');
    
    [status,message,messageid] = mkdir(p.test_subdir_path);
    
end

end

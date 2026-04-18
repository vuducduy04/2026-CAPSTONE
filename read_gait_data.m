data_root = 'C:\Users\Vu Duc Duy\Desktop\Sem 8\CAPSTONE\Gait120';
subjects = {'S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 
            'S009', 'S010', 'S011', 'S012', 'S013', 'S014', 'S015', 'S016', 
            'S017', 'S018', 'S019', 'S020', 'S021', 'S022', 'S023', 'S024', 
            'S025', 'S026', 'S027', 'S028', 'S029', 'S030', 'S031', 'S032', 
            'S033', 'S034', 'S035', 'S036', 'S037', 'S038', 'S039', 'S040', 
            'S041', 'S042', 'S043', 'S044', 'S045', 'S046', 'S047', 'S048', 
            'S049', 'S050', 'S051', 'S052', 'S053', 'S054', 'S055', 'S056', 
            'S057', 'S058', 'S059', 'S060', 'S061', 'S062', 'S063', 'S064', 
            'S065', 'S066', 'S067', 'S068', 'S069', 'S070', 'S071', 'S072', 
            'S073', 'S074', 'S075', 'S076', 'S077', 'S078', 'S079', 'S080', 
            'S081', 'S082', 'S083', 'S084', 'S085', 'S086', 'S087', 'S088', 
            'S089', 'S090', 'S091', 'S092', 'S093', 'S094', 'S095', 'S096', 
            'S097', 'S098', 'S099', 'S100', 'S101', 'S102', 'S103', 'S104', 
            'S105', 'S106', 'S107', 'S108', 'S109', 'S110', 'S111', 'S112', 
            'S113', 'S114', 'S115', 'S116', 'S117', 'S118', 'S119', 'S120'};
% subjects = {'S001'};
task_name = 'LevelWalking';
muscle_idx = [1, 4, 6, 7]; % 1-indexed: VL, TA, ST, GM

output_dir = fullfile(data_root, 'output');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for s = 1:numel(subjects)
    mat = load(fullfile(data_root, subjects{s}, 'EMG', 'ProcessedData.mat'));
    task = mat.(task_name);
    for tri = task.AvailableTrialIdx
        trial_name = sprintf('Trial%02d', tri);
        trial = task.(trial_name);
        for step = 1:trial.nSteps
            step_name = sprintf('Step%02d', step);
            emg = table2array(trial.(step_name).EMGs_interpolated);
            % emg is [101 × 12], each column = one muscle
            vl = emg(:, 1);   % Vastus Lateralis
            ta = emg(:, 4);   % Tibialis Anterior
            st = emg(:, 6);   % Semitendinosus
            gm = emg(:, 7);   % Gastrocnemius Medialis
            % Save to CSV
            fname = sprintf('%s_%s_%s.csv', subjects{s}, trial_name, step_name);
            writematrix([vl, ta, st, gm], fullfile(output_dir, fname));
        end
    end
end
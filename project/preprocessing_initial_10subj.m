clear;
clc;
eeglab;

% subject list
% subjects = {'sub-001', 'sub-002', 'sub-003', 'sub-004', 'sub-005', ...
            % 'sub-006', 'sub-007', 'sub-008', 'sub-009', 'sub-011'}; 
subjects = 'sub-001';


all_num_trials = zeros(1, length(subjects));
rejection_metadata = table('Size', [length(subjects) 8], ...
    'VariableTypes', {'string', 'double', 'double', 'double', 'double', 'double', 'double', 'cell'}, ...
    'VariableNames', {'Subject', 'TotalChannels', 'RejectedChannels', ...
                      'FinalChannels', 'TotalTrials', 'RejectedTrials', 'FinalTrials', 'GoodTrialIndices'});

for subj_idx = 1:length(subjects)
    subj_id = subjects{subj_idx};
    fprintf('Processing %s...\n', subj_id);
    edf_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_eeg.edf', subj_id, subj_id);
    event_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_events.tsv', subj_id, subj_id);
    output_dir = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/spectral_data', subj_id);
    
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Load EEG data
    try
        EEG = pop_biosig(edf_file);
    catch ME
        warning('Could not load subject %s: %s', subj_id, ME.message);
        continue;
    end
    
    % insert events
    event_data = readtable(event_file, 'FileType', 'text', 'Delimiter', '\t');
    EEG.event = struct([]);
    for i = 1:height(event_data)
        EEG.event(i).latency = event_data.sample(i);
        EEG.event(i).type = string(event_data.trial_type(i));
        EEG.event(i).duration = event_data.duration(i) * EEG.srate;
    end
    EEG = eeg_checkset(EEG, 'eventconsistency');
    
    chan_labels = {EEG.chanlocs.labels};
    
    % Spectral plots (initial inspection) 
    figure;
    [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
    saveas(gcf, fullfile(output_dir, 'spectro_raw.png'), 'png');
    close(gcf); % Close the figure
    
    % Notch filter (50 Hz)
    wo = 50/(EEG.srate/2); bo = wo/35;
    [bn,an] = iirnotch(wo, bo);
    EEG.data = filtfilt(bn, an, EEG.data')';
    EEG = eeg_checkset(EEG);
    figure;
    [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
    saveas(gcf, fullfile(output_dir, 'spectro_notch.png'), 'png');
    close(gcf);
    
    % Epoch extraction
    types = {EEG.event.type};
    start_idx = find(strcmp(types, 'trialStart'));
    end_idx = find(strcmp(types, 'trialEnd'));
    pre_buffer = round(-0.2 * EEG.srate); % 200 ms pre-stimulus
    post_buffer = round(0.8 * EEG.srate); % 800 ms post-stimulus
    num_trials = min(length(start_idx), length(end_idx));
    epochs = cell(1, num_trials);
    for t = 1:num_trials
        start_lat = round(EEG.event(start_idx(t)).latency) + pre_buffer;
        end_lat = round(EEG.event(end_idx(t)).latency) + post_buffer;
        if start_lat < 1 || end_lat > size(EEG.data, 2) || start_lat >= end_lat
            warning('Skipping trial %d: start=%d, end=%d', t, start_lat, end_lat);
            continue;
        end
        epochs{t} = EEG.data(:, start_lat:end_lat);
    end

    % Filter out empty epochs
    epochs = epochs(~cellfun('isempty', epochs));
    num_trials = length(epochs);

    % Downsample
    new_srate = 512;
    for t = 1:num_trials
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN before downsampling, skipping.', t);
            continue;
        end
        epochs{t} = resample(epochs{t}', new_srate, EEG.srate)';
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_downsampled.png'), 'png');
    close(gcf);
    
    % Bandpass filter (0.5-40 Hz)
    for t = 1:num_trials
        if any(isnan(epochs{t}(:)))
            warning('Trial %d contains NaN before filtering, skipping.', t);
            continue;
        end
        [b,a] = butter(2, [0.5 40]/(new_srate/2), 'bandpass');
        epochs{t} = filtfilt(b, a, epochs{t}')';
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_bandpass.png'), 'png');
    close(gcf);
    
    % Manual bad channels rejection
    manual_bad_indices = {
        'sub-001', [20]; 'sub-002', [20, 2, 3]; 'sub-003', [57, 60];
        'sub-004', [57]; 'sub-005', [20]; 'sub-006', [31, 30];
        'sub-007', [18, 19, 27]; 'sub-008', [31]; 'sub-009', [20, 26, 35];
        'sub-011', [52, 61, 64, 31]
    };
    bad_indices = [];
    for i = 1:size(manual_bad_indices, 1)
        if strcmp(manual_bad_indices{i, 1}, subj_id)
            bad_indices = manual_bad_indices{i, 2};
            break;
        end
    end
    
    % Channel Rejection
    num_channels = size(epochs{1}, 1);
    max_samples = max(cellfun(@(x) size(x, 2), epochs));
    epochs_3d = zeros(num_channels, max_samples, num_trials);
    for t = 1:num_trials
        epochs_3d(:,:,t) = [epochs{t}, zeros(num_channels, max_samples - size(epochs{t}, 2))];
    end
    
    chan_var = squeeze(var(epochs_3d, 0, [2 3]));
    bad_chans_auto = (chan_var < 1e-6) | (chan_var > mean(chan_var) + 3 * std(chan_var));
    bad_chans_auto(bad_indices) = true;
    good_chans = find(~bad_chans_auto);
    num_bad_chans = sum(bad_chans_auto);
    num_total_chans = length(chan_labels);
    
    if num_bad_chans > 0
        rejected_indices = find(bad_chans_auto);
        rejected_labels = chan_labels(rejected_indices);
        fprintf('Rejected %d/%d channels (labels: %s)\n', num_bad_chans, num_total_chans, strjoin(rejected_labels, ', '));
    else
        fprintf('Rejected %d/%d channels\n', num_bad_chans, num_total_chans);
    end
    
    epochs = cell(1, num_trials);
    for t = 1:num_trials
        epochs{t} = epochs_3d(good_chans, :, t);
    end
    chan_labels_clean = chan_labels(good_chans);
    
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_channel_rejected.png'), 'png');
    close(gcf);
    
    % Common Average Reference (CAR)
    for t = 1:num_trials
        mean_signal = mean(epochs{t}, 1);
        epochs{t} = epochs{t} - repmat(mean_signal, size(epochs{t}, 1), 1);
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_car.png'), 'png');
    close(gcf);
    
    % Subtract Baseline
    baseline_window = [-0.2 0];
    for t = 1:num_trials
        baseline_samples = round((baseline_window(1) + 0.2) * new_srate : (baseline_window(2) + 0.2) * new_srate);
        baseline_samples = max(1, min(size(epochs{t}, 2), baseline_samples));
        baseline_mean = mean(epochs{t}(:, baseline_samples), 2);
        epochs{t} = epochs{t} - repmat(baseline_mean, 1, size(epochs{t}, 2));
    end
    temp_data = cat(2, epochs{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_baseline.png'), 'png');
    close(gcf);
    
    % Trial Rejection per subject
    max_samples = max(cellfun(@(x) size(x, 2), epochs));
    epochs_3d = zeros(size(epochs{1}, 1), max_samples, num_trials);
    for t = 1:num_trials
        epochs_3d(:,:,t) = [epochs{t}, zeros(size(epochs{t}, 1), max_samples - size(epochs{t}, 2))];
    end
    
    trial_var = squeeze(var(epochs_3d, 0, 2));
    trial_mean_var = mean(trial_var, 1);
    subject_mean_var = mean(trial_mean_var);
    subject_std_var = std(trial_mean_var);
    trial_reject = trial_mean_var > (subject_mean_var + 3 * subject_std_var);
    
    good_trials = ~trial_reject;
    epochs_3d = epochs_3d(:,:,good_trials);
    fprintf('Rejected %d/%d trials\n', sum(trial_reject), length(trial_reject));
    
    epochs_final = cell(1, sum(good_trials));
    for t = 1:sum(good_trials)
        epochs_final{t} = epochs_3d(:,:,t);
        non_zero_idx = find(any(epochs_final{t} ~= 0, 1), 1, 'last');
        if ~isempty(non_zero_idx)
            epochs_final{t} = epochs_final{t}(:, 1:non_zero_idx);
        end
    end

    old_num_trials = num_trials;
    num_trials_final = sum(good_trials);
    all_num_trials(subj_idx) = num_trials_final;
    
    temp_data = cat(2, epochs_final{:});
    figure;
    [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
    saveas(gcf, fullfile(output_dir, 'spectro_trial_reject.png'), 'png');
    close(gcf);
    
    good_trial_idx = find(good_trials); 
    
    % Update metadata table
    rejection_metadata.Subject(subj_idx) = subj_id;
    rejection_metadata.TotalChannels(subj_idx) = num_total_chans;
    rejection_metadata.RejectedChannels(subj_idx) = num_bad_chans;
    rejection_metadata.FinalChannels(subj_idx) = length(good_chans);
    rejection_metadata.TotalTrials(subj_idx) = old_num_trials;
    rejection_metadata.RejectedTrials(subj_idx) = sum(trial_reject);
    rejection_metadata.FinalTrials(subj_idx) = num_trials_final;
    rejection_metadata.GoodTrialIndices{subj_idx} = good_trial_idx;
    
    save(fullfile(output_dir, 'pre_ica_epochs.mat'), 'epochs_final', 'new_srate', 'chan_labels_clean');
    
end

fprintf('Preprocessing completed for all subjects.\n');

% Bar chart of number of trials per subject
figure;
bar(all_num_trials, 'FaceColor', [0.2 0.6 0.8]);
xticks(1:length(subjects));
xticklabels(subjects);
xtickangle(45);
xlabel('Subjects');
ylabel('Number of Trials');
title('Number of Trials per Subject');
grid on;
saveas(gcf, fullfile('/home/ozoswita/Dataset/Essex_Movie/output', 'trials_per_subject.png'), 'png');

% save all rejection metadata in a csv
metadata_file = '/home/ozoswita/Dataset/Essex_Movie/output/rejection_metadata.csv';
rejection_metadata.GoodTrialIndices = cellfun(@mat2str, rejection_metadata.GoodTrialIndices, 'UniformOutput', false);
writetable(rejection_metadata, metadata_file);
fprintf('Rejection metadata saved to %s\n', metadata_file);

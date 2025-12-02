%% Full Pipeline for 1 subject

clear; clc;
rng(12345, 'twister');
eeglab; close all;

subj_id = 'sub-007';
edf_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_eeg.edf', subj_id, subj_id);
event_file = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/eeg/%s_task-MovieMemory_events.tsv', subj_id, subj_id);
output_dir = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/unit_test_results', subj_id);
erp_output_dir = sprintf('/home/ozoswita/Dataset/Essex_Movie/%s/unit_test_results/erp_results', subj_id);
if ~exist(output_dir, 'dir'), mkdir(output_dir); end
if ~exist(erp_output_dir, 'dir'), mkdir(erp_output_dir); end

fprintf('\n RUNNING FULL PIPELINE FOR %s: \n', subj_id);

%% Preprocessing

% Load EEG data
try
    EEG = pop_biosig(edf_file);
catch ME
    error('Could not load subject %s: %s', subj_id, ME.message);
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
% figure;
% [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
% saveas(gcf, fullfile(output_dir, 'spectro_raw.png'), 'png');
% close(gcf);

% Notch filter (50 Hz) 
wo = 50/(EEG.srate/2); bo = wo/35;
[bn,an] = iirnotch(wo, bo);
EEG.data = filtfilt(bn, an, EEG.data')';
EEG = eeg_checkset(EEG);
% figure;
% [~, ~, ~, ~] = spectopo(EEG.data, 0, EEG.srate, 'freqrange', [0 100]);
% saveas(gcf, fullfile(output_dir, 'spectro_notch.png'), 'png');
% close(gcf);

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

% Downsample to 512 hz
new_srate = 512;
for t = 1:num_trials
    epochs{t} = resample(epochs{t}', new_srate, EEG.srate)';
end
temp_data = cat(2, epochs{:});
figure;
[~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
saveas(gcf, fullfile(output_dir, 'spectro_downsampled.png'), 'png');
close(gcf);

%% check slow drifts before bandpass
slow_drift_channel = 11; % picking a random channel
temp_data = cat(2, epochs{:});

figure('Name',[subj_id ' - Slow Drift BEFORE Bandpass'],'Color','w');
plot((1:size(temp_data,2))/new_srate, temp_data(slow_drift_channel,:));
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(sprintf('%s - Channel %s (Before Bandpass)', subj_id, chan_labels{slow_drift_channel}));
xlim([0 size(temp_data,2)/new_srate]);
grid on;

% Bandpass filter (1–25 Hz)
[b,a] = butter(2, [1 25]/(new_srate/2), 'bandpass');
for t = 1:num_trials
    epochs{t} = filtfilt(b, a, epochs{t}')';
end

%% check slow drifts AFTER bandpass
temp_data = cat(2, epochs{:});
figure('Name',[subj_id ' - Slow Drift AFTER Bandpass'],'Color','w');
plot((1:size(temp_data,2))/new_srate, temp_data(slow_drift_channel,:));
xlabel('Time (s)');
ylabel('Amplitude (\muV)');
title(sprintf('%s - Channel %s (After Bandpass)', subj_id, chan_labels{slow_drift_channel}));
xlim([0 size(temp_data,2)/new_srate]);
grid on;

temp_data = cat(2, epochs{:});
figure;
[~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
saveas(gcf, fullfile(output_dir, 'spectro_bandpass.png'), 'png');
close(gcf);

% removing bad channels manually

% manual_bad_indices = {
%     'sub-001', [20]; 'sub-002', [20, 2, 3]; 'sub-003', [57, 60];
%     'sub-004', [57]; 'sub-005', [20]; 'sub-006', [31, 30];
%     'sub-007', [18, 19, 27]; 'sub-008', [31]; 'sub-009', [20, 26, 35];
%     'sub-011', [52, 61, 64, 31]
% };

bad_indices = [18, 19, 27]; % subject -07

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
    rejected_labels = chan_labels(bad_chans_auto);
    fprintf('Rejected %d/%d channels (labels: %s)\n', num_bad_chans, num_total_chans, strjoin(rejected_labels, ', '));
else
    fprintf('Rejected %d/%d channels\n', num_bad_chans, num_total_chans);
end

epochs = cell(1, num_trials);
for t = 1:num_trials
    epochs{t} = epochs_3d(good_chans, :, t);
end
chan_labels_clean = chan_labels(good_chans);

% temp_data = cat(2, epochs{:});
% figure;
% [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
% saveas(gcf, fullfile(output_dir, 'spectro_channel_rejected.png'), 'png');
% close(gcf);

% Common Average Reference
for t = 1:num_trials
    mean_signal = mean(epochs{t}, 1);
    epochs{t} = epochs{t} - mean_signal;
end
% temp_data = cat(2, epochs{:});
% figure;
% [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
% saveas(gcf, fullfile(output_dir, 'spectro_car.png'), 'png');
% close(gcf);

% % Baseline Subtraction –0.2 to 0 s
% baseline_window = [-0.2 0];
% time_vector = linspace(-0.2, 0.8, size(epochs{1}, 2));  % consistent with epoching window
% 
% % Find indices corresponding to the baseline window
% baseline_samples = find(time_vector >= baseline_window(1) & time_vector <= baseline_window(2));
% 
% for t = 1:num_trials
%     if isempty(baseline_samples)
%         warning('No valid baseline samples for trial %d', t);
%         continue;
%     end
%     baseline_mean = mean(epochs{t}(:, baseline_samples), 2);
%     epochs{t} = epochs{t} - baseline_mean;
% end

% temp_data = cat(2, epochs{:});
% figure;
% [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
% saveas(gcf, fullfile(output_dir, 'spectro_baseline.png'), 'png');
% close(gcf);

% Trial Rejection (variance > mean + 3SD)
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

num_trials_final = sum(good_trials);
fprintf('Final: %d good trials remaining.\n', num_trials_final);

% temp_data = cat(2, epochs_final{:});
% figure;
% [~, ~, ~, ~] = spectopo(temp_data, 0, new_srate, 'freqrange', [0 50]);
% saveas(gcf, fullfile(output_dir, 'spectro_trial_reject.png'), 'png');
% close(gcf);


%% ICA

fprintf('\n Running ICA:\n');

% Combine all epochs into continuous data
data_ica = cat(2, epochs_final{:});

num_chans = length(chan_labels_clean);
chanlocs_struct = struct('labels', chan_labels_clean);
eeglabpath = fileparts(which('eeglab.m'));
chanlocs_std = readlocs(fullfile(eeglabpath, 'plugins', 'dipfit', ...
                                'standard_BEM', 'elec', 'standard_1005.elc'));
for c = 1:num_chans
    idx_std = find(strcmpi({chanlocs_std.labels}, chan_labels_clean{c}));
    if ~isempty(idx_std)
        chanlocs_struct(c).X = chanlocs_std(idx_std).X;
        chanlocs_struct(c).Y = chanlocs_std(idx_std).Y;
        chanlocs_struct(c).Z = chanlocs_std(idx_std).Z;
    else
        chanlocs_struct(c).X = NaN;
        chanlocs_struct(c).Y = NaN;
        chanlocs_struct(c).Z = NaN;
    end
end

EEG_ica = pop_importdata('data', data_ica, 'setname', [subj_id '_ICA'], ...
                         'srate', new_srate, 'chanlocs', chanlocs_struct);
EEG_ica = eeg_checkset(EEG_ica);

rng(12345, 'twister');  % random seed for reproducibility

rankEEG = rank(double(EEG_ica.data'));  

fprintf('Using PCA rank = %d\n', rankEEG);

% ICA with PCA reduction
EEG_ica = pop_runica(EEG_ica, 'extended', 1, 'pca', rankEEG, 'interrupt', 'on');

EEG_ica.icaact = (EEG_ica.icaweights * EEG_ica.icasphere) * EEG_ica.data;
fprintf('ICA completed with %d components.\n', size(EEG_ica.icaweights,1));

% check topo plot for first 10 ICs
figure;
pop_topoplot(EEG_ica, 0, 1:10, [subj_id ' - Top 10 ICA Components'], 0, 'electrodes', 'on');

numICs = size(EEG_ica.icaweights, 1);
plotsPerFig = 5;
for startIC = 1:plotsPerFig:numICs
    figure;
    for k = 1:plotsPerFig
        ic = startIC + k - 1;
        if ic > numICs, break; end
        subplot(plotsPerFig, 2, (k-1)*2 + 1);
        plot(EEG_ica.icaact(ic,:));
        xlabel('Samples'); ylabel(['IC ' num2str(ic)]);
        title(['IC ' num2str(ic) ' Time Series']);
        subplot(plotsPerFig, 2, (k-1)*2 + 2);
        [pxx,f] = pwelch(EEG_ica.icaact(ic,:), [], [], [], EEG_ica.srate);
        plot(f,10*log10(pxx)); xlim([1 40]);
        xlabel('Freq (Hz)'); ylabel('Power (dB)');
        title(['IC ' num2str(ic) ' Spectrum']);
    end
    sgtitle([subj_id ' - ICs ' num2str(startIC) ' to ' ...
             num2str(min(startIC+plotsPerFig-1,numICs))]);
end

% remove bad IC components manually

% remove_comps = [2, 4, 6, 8, 12, 15, 24, 62]; % subject 1
remove_comps = [5, 6, 7, 8, 11, 17, 25, 31, 32, 33, 45, 59]; % subject 2
% remove_comps = [2, 3, 5, 6, 8, 11, 12, 13, 14, 16, 17, 19, 20, 21, 22, 24, 27, 28, 54, 58, 62]; % subject 3
% remove_comps = [3, 4, 10, 17, 20, 40, 50, 57, 59]; % subject 4
% remove_comps = [1, 4, 5, 7, 8, 10, 13, 19, 20, 21, 22, 35, 59, 62]; % subject 5
% remove_comps = [1, 3, 5, 7, 8, 9, 10, 14, 16, 17, 25, 29, 44, 61]; % subject 6
% remove_comps = [1, 2, 4, 6, 8, 9, 10, 11, 17, 21, 31, 61]; % subject 7
% remove_comps = [1, 2, 3, 4, 7, 9, 13, 16, 19, 20, 23, 32, 34, 38, 40, 42, 45]; % subject 8
% remove_comps = [2, 5, 9, 10, 11, 12, 13, 15, 25, 33, 35, 47, 49, 51, 56]; % subject 9
% remove_comps = [1, 3, 5, 7, 9, 11, 14, 15, 18, 28, 58]; % subject 11
remove_comps(remove_comps > size(EEG_ica.icaweights,1)) = [];
if ~isempty(remove_comps)
    EEG_ica = pop_subcomp(EEG_ica, remove_comps, 0);
    fprintf('Removed components: %s\n', mat2str(remove_comps));
end

fprintf('\nICA complete for %s\n', subj_id);

fprintf('FC1 channel index: %d\n', find(strcmpi({EEG_ica.chanlocs.labels}, 'FC1')));
fc1_idx = find(strcmpi({EEG_ica.chanlocs.labels}, 'FC1'));
figure;
plot(EEG_ica.data(fc1_idx,1:EEG_ica.srate*30)); % first 30 seconds
title('FC1 After Manual ICA Cleanup');
xlabel('Time (s)'); ylabel('Amplitude (\muV)');

% post-ICA cleaned dataset
post_ica_file = fullfile(output_dir, [subj_id '_postICA.set']);
pop_saveset(EEG_ica, 'filename', [subj_id '_postICA.set'], 'filepath', output_dir);
fprintf('Saved post-ICA cleaned dataset: %s\n', post_ica_file);

%%  Sort Trials into R / K / M memory conditins

post_ica_file = fullfile(output_dir, [subj_id '_postICA.set']);

if exist(post_ica_file, 'file')
    fprintf('\nFound post-ICA dataset — loading it\n');
    EEG_ica = pop_loadset('filename', [subj_id '_postICA.set'], 'filepath', output_dir);
    fprintf('Loaded: %s\n', post_ica_file);
else
    error('No post-ICA dataset found. Please run ICA step first.');
end

fprintf('\n Sorting trials into conditions (R/K/M) \n');

% Event types corresponding to clip starts
event_types = {'startOfNotRecognisedClip', 'startOfRememberedClipFirstWatch', 'startOfRecognisedClipFirstWatch'};

if ~exist('event_data','var')
    error('event_data not found. Please ensure events were loaded earlier.');
end

% finding shortest clip duration across event types
min_duration_samples = Inf;
for i = 1:height(event_data)
    if ismember(event_data.trial_type{i}, event_types)
        clip_start_sample = event_data.sample(i);
        end_idx = find(strcmp(event_data.trial_type, 'endOfClip') & event_data.sample > clip_start_sample, 1, 'first');
        if ~isempty(end_idx)
            duration = event_data.sample(end_idx) - clip_start_sample;
            if duration < min_duration_samples
                min_duration_samples = duration;
            end
        end
    end
end

if isinf(min_duration_samples)
    warning('No valid clip durations found.');
    return;
end

min_duration_samples_new_srate = round(min_duration_samples / 2048 * new_srate);
baseline_length = round(0.4 * new_srate);
fprintf('Shortest clip duration: %d (orig) → %d (resampled)\n', ...
    min_duration_samples, min_duration_samples_new_srate);

all_epochs_struct_full = struct('remembered', [], 'recognised', [], 'not_recognised', []);

for i = 1:height(event_data)
    current_event_type = event_data.trial_type{i};

    if ismember(current_event_type, event_types)
        % Find the last trialStart before this clip start
        trial_start_event_idx = find(strcmp(event_data.trial_type, 'trialStart') & ...
            event_data.sample < event_data.sample(i), 1, 'last');
        if isempty(trial_start_event_idx), continue; end

        % Find corresponding trial number
        trial_number = sum(strcmp(event_data.trial_type(1:trial_start_event_idx), 'trialStart'));
        if trial_number > length(epochs_final)
            continue;
        end

        offset_samples_orig = event_data.sample(i) - event_data.sample(trial_start_event_idx);

        % epoching with hypothesis 1 specific baseline

        baseline_sec = 0.4;  % in sec
        baseline_samples = round(baseline_sec * new_srate);

        % start index relative to trialStart (converted from 2048 → 512 Hz)
        start_idx_new = round(offset_samples_orig / 2048 * new_srate) + baseline_length + 1 - baseline_samples;

        % end index includes full clip duration + baseline window
        end_idx_new = start_idx_new + min_duration_samples_new_srate - 1 + baseline_samples;

        trial_lat_start = round(EEG.event(start_idx(trial_number)).latency / (EEG.srate / new_srate));
        trial_lat_end   = trial_lat_start + size(epochs_final{trial_number},2) - 1;

        % Convert cleaned data to match trial
        if trial_lat_start < 1 || trial_lat_end > size(EEG_ica.data,2)
            warning('Trial %d out of ICA data bounds, skipping.', trial_number);
            continue;
        end
        trial_data_clean = EEG_ica.data(:, trial_lat_start:trial_lat_end);

        if start_idx_new < 1 || end_idx_new > size(trial_data_clean, 2)
            warning('Invalid indices for trial %d, skipping.', trial_number);
            continue;
        end

        new_epoch_data = trial_data_clean(:, start_idx_new:end_idx_new);

        % Assign condition
        if contains(current_event_type, 'NotRecognised', 'IgnoreCase', true)
            cond = 'not_recognised';
        elseif contains(current_event_type, 'Remembered', 'IgnoreCase', true)
            cond = 'remembered';
        elseif contains(current_event_type, 'Recognised', 'IgnoreCase', true)
            cond = 'recognised';
        else
            continue;
        end

        if isempty(all_epochs_struct_full.(cond))
            all_epochs_struct_full.(cond) = new_epoch_data;
        else
            all_epochs_struct_full.(cond) = cat(3, all_epochs_struct_full.(cond), new_epoch_data);
        end
    end
end

fprintf('Extracted ICA-cleaned epochs:\n');
fprintf('  Remembered: %d\n', size(all_epochs_struct_full.remembered,3));
fprintf('  Recognised: %d\n', size(all_epochs_struct_full.recognised,3));
fprintf('  Not Recognised: %d\n', size(all_epochs_struct_full.not_recognised,3));

% Balance trial counts across conditions
fields = {'remembered', 'recognised', 'not_recognised'};
min_epochs = min(cellfun(@(f) size(all_epochs_struct_full.(f), 3), fields));
for f_idx = 1:length(fields)
    f = fields{f_idx};
    num = size(all_epochs_struct_full.(f), 3);
    if num > min_epochs
        rng(12345, 'twister');
        idx = randperm(num, min_epochs);
        all_epochs_struct_full.(f) = all_epochs_struct_full.(f)(:,:,idx);
    end
end

fprintf('Balanced all ICA-cleaned conditions to %d trials each.\n', min_epochs);

% copy of the pre-baseline data for TF analysis 
all_epochs_struct_raw = all_epochs_struct_full;  

%% Baseline correction for H1 epochs (−0.4 to 0 s)
fprintf('\n Applying baseline correction (−0.4 to 0 s)\n');

% Recreate time vector for each epoch (starts at −0.4 s)
time_vector = linspace(-0.4, (size(all_epochs_struct_full.remembered, 2)/new_srate) - 0.4, size(all_epochs_struct_full.remembered, 2));

baseline_window = [-0.4 0];
baseline_idx = find(time_vector >= baseline_window(1) & time_vector <= baseline_window(2));

% Apply baseline correction per condition
for cond = {'remembered','recognised','not_recognised'}
    data = all_epochs_struct_full.(cond{1});
    if isempty(data)
        continue;
    end
    fprintf('Applying baseline correction for %s (%d trials)\n', cond{1}, size(data,3));
    for t = 1:size(data,3)
        baseline_mean = mean(data(:, baseline_idx, t), 2);
        data(:,:,t) = data(:,:,t) - baseline_mean;
    end
    all_epochs_struct_full.(cond{1}) = data;
end
fprintf('Baseline correction complete.\n\n');


%% ERP ANALYSIS


fprintf('\n Running ERP Analysis (Per Channel)\n');

% Load per condition data
epochs_remembered     = all_epochs_struct_full.remembered;
epochs_recognised     = all_epochs_struct_full.recognised;
epochs_not_recognised = all_epochs_struct_full.not_recognised;

% channels of interest 

chan_names_of_interest = {'Fz', 'FCz', 'AFz', 'F1', 'F2', 'FC1', 'FC2', 'P3', 'CP3', 'P1', 'P5', 'CP1', 'PO3', 'O1', 'Oz', 'O2'};
all_labels = {EEG_ica.chanlocs.labels};

% Match and find actual EEG indices
chan_indices = [];
chan_names_found = {};
for n = 1:length(chan_names_of_interest)
    idx = find(strcmpi(all_labels, chan_names_of_interest{n}));
    if ~isempty(idx)
        chan_indices(end+1) = idx;
        chan_names_found{end+1} = chan_names_of_interest{n};
    else
        warning('Channel %s not found in EEG data.', chan_names_of_interest{n});
    end
end

if isempty(chan_indices)
    error('No valid channels found for ERP plotting.');
end

fprintf('\n Channel index check \n');
for idx = chan_indices
    fprintf('Index %d → %s\n', idx, EEG_ica.chanlocs(idx).labels);
end

fprintf('Found channels: %s\n', strjoin(chan_names_found, ', '));

time_points = linspace(-0.4, (size(epochs_remembered, 2)/new_srate) - 0.4, size(epochs_remembered, 2));


% Compute ERP per condition (mean across trials, per channel) 
erp_R = squeeze(mean(epochs_remembered(chan_indices,:,:), 3));  % Remembered
erp_K = squeeze(mean(epochs_recognised(chan_indices,:,:), 3));  % Recognised
erp_M = squeeze(mean(epochs_not_recognised(chan_indices,:,:), 3));  % Not Recognised

if isvector(erp_R)
    erp_R = erp_R(:)'; erp_K = erp_K(:)'; erp_M = erp_M(:)';
end

%% ERP PLOTS
fprintf('\n ERP plotting \n');

cR = [0 0.45 0.74];  % blue
cK = [0.85 0.33 0.1]; % red
cM = [0.47 0.67 0.19]; % green

% reference time lines will be at (in seconds)
ref_times = [0.1 0.3 0.5 0.8];

num_chans = length(chan_names_found);

for idx = 1:2:num_chans
    chanA = idx;
    chanB = idx + 1;

    if chanB <= num_chans
        fig_title = sprintf('ERP Channels %s & %s', chan_names_found{chanA}, chan_names_found{chanB});
        save_name = sprintf('ERP_%s_%s', chan_names_found{chanA}, chan_names_found{chanB});
    else
        fig_title = sprintf('ERP Channel %s', chan_names_found{chanA});
        save_name = sprintf('ERP_%s', chan_names_found{chanA});
    end

    fig = figure('Name', fig_title, 'Position', [200 200 1000 600], 'Color', 'w');

    subplot(2,1,1);
    plot(time_points, erp_R(chanA,:), 'Color', cR, 'LineWidth', 1.5); hold on;
    plot(time_points, erp_K(chanA,:), 'Color', cK, 'LineWidth', 1.5);
    plot(time_points, erp_M(chanA,:), 'Color', cM, 'LineWidth', 1.5);
    line([0 0], ylim, 'Color','k', 'LineStyle','--');

    for rt = ref_times
        line([rt rt], ylim, 'Color',[0.4 0.4 0.4], 'LineStyle','--');
    end

    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    title(sprintf('Channel: %s', chan_names_found{chanA}));
    legend({'Remembered','Recognised','Not Recognised'}, 'Location', 'best');
    grid on; xlim([-0.4 0.8]);

    if chanB <= num_chans
        subplot(2,1,2);
        plot(time_points, erp_R(chanB,:), 'Color', cR, 'LineWidth', 1.5); hold on;
        plot(time_points, erp_K(chanB,:), 'Color', cK, 'LineWidth', 1.5);
        plot(time_points, erp_M(chanB,:), 'Color', cM, 'LineWidth', 1.5);
        line([0 0], ylim, 'Color','k', 'LineStyle','--');

        for rt = ref_times
            line([rt rt], ylim, 'Color',[0.4 0.4 0.4], 'LineStyle','--');
        end

        xlabel('Time (s)');
        ylabel('Amplitude (µV)');
        title(sprintf('Channel: %s', chan_names_found{chanB}));
        legend({'Remembered','Recognised','Not Recognised'}, 'Location', 'best');
        grid on; xlim([-0.4 0.8]);
    end

    saveas(fig, fullfile(erp_output_dir, [save_name '.png']));
    savefig(fig, fullfile(erp_output_dir, [save_name '.fig']));
    close(fig);

    fprintf('Saved ERP figure: %s\n', fullfile(erp_output_dir, [save_name '.png']));
end

fprintf('\nAll ERP plots saved to: %s\n\n', erp_output_dir);

%%  Difference Waves (R-K, K-M)

fprintf('\n Difference Wave Analysis (R-K and K-M): \n');

ref_times = [0.1 0.3 0.5 0.8];

num_chans = length(chan_names_found);

diff_output_dir = fullfile(erp_output_dir, 'difference_waves');
if ~exist(diff_output_dir, 'dir'), mkdir(diff_output_dir); end

for c = 1:num_chans
    
    chan_name = chan_names_found{c};
    
    % Difference waves
    diff_RK = erp_R(c,:) - erp_K(c,:);
    diff_KM = erp_K(c,:) - erp_M(c,:);
    
    fig = figure('Name', ['Diff ' chan_name], ...
                 'Position', [200 200 1000 500], 'Color', 'w');
    
    % R - K 
 
    subplot(2,1,1)
    plot(time_points, diff_RK, 'LineWidth', 1.5, 'Color', [0.1 0.4 0.7]); hold on;
    line([0 0], ylim, 'Color','k','LineStyle','--');  
    
    for rt = ref_times
        line([rt rt], ylim, 'Color', [0.4 0.4 0.4], 'LineStyle','--');
    end
    
    title(sprintf('R - K (Recollection) | %s', chan_name));
    ylabel('Amplitude (µV)');
    grid on;
    xlim([-0.4 0.8]);
   
    % K - M

    subplot(2,1,2)
    plot(time_points, diff_KM, 'LineWidth', 1.5, 'Color', [0.85 0.33 0.1]); hold on;
    line([0 0], ylim, 'Color','k','LineStyle','--'); 
    
    for rt = ref_times
        line([rt rt], ylim, 'Color', [0.4 0.4 0.4], 'LineStyle','--');
    end
    
    title(sprintf('K - M (Familiarity) | %s', chan_name));
    xlabel('Time (s)');
    ylabel('Amplitude (µV)');
    grid on;
    xlim([-0.4 0.8]);
    
    saveas(fig, fullfile(diff_output_dir, ['Diff_' chan_name '.png']));
    savefig(fig, fullfile(diff_output_dir, ['Diff_' chan_name '.fig']));
    close(fig);
    
    fprintf('Saved difference wave for channel %s\n', chan_name);
end

fprintf('\nDifference wave plots saved to: %s\n\n', diff_output_dir);

%%  ERP Stats


fprintf('\n ERP Statistics: \n');

% Time windows 
win_names = {'FN400','LPC'};
win_times = [0.15 0.45;   % FN400
             0.50 0.80];  % LPC

% Convert to indices
win_idx = cell(size(win_names));
for w = 1:numel(win_names)
    win_idx{w} = find(time_points >= win_times(w,1) & ...
                      time_points <= win_times(w,2));
end

% ROI definitions 
roi_def_erp = struct();
roi_def_erp.MidFrontal   = {'Fz','FCz','AFz','F1','F2','FC1','FC2'};
roi_def_erp.LeftParietal = {'P3','CP3','P1','P5','CP1','PO3'};
roi_def_erp.ControlOccipital = {'O1','Oz','O2'};

roi_names_erp = fieldnames(roi_def_erp);

all_labels_full = {EEG_ica.chanlocs.labels};

nROI = numel(roi_names_erp);
nWin = numel(win_names);
p_main  = nan(nROI,nWin);   % main effect
p_RK    = nan(nROI,nWin);   % R-K (recollection)
p_KM    = nan(nROI,nWin);   % K-M (familiarity)

erp_roi_stats = struct();

stats_file = fullfile(output_dir, sprintf('%s_ERP_ROI_stats.txt', subj_id));
fid = fopen(stats_file, 'w');
fprintf(fid, 'ERP ROI Statistical Results for %s\n', subj_id);
fprintf(fid, 'Windows: FN400 = 150–450 ms, LPC = 500–800 ms\n');
fprintf(fid, 'Condition factor: R (Remembered), K (Recognised), M (Not recognised)\n');

for r = 1:nROI

    roi_label = roi_names_erp{r};
    roi_chans = roi_def_erp.(roi_label);

    roi_idx = find(ismember(all_labels_full, roi_chans));

    if isempty(roi_idx)
        fprintf('ROI %s: no matching channels found, skipping.\n', roi_label);
        fprintf(fid, 'ROI %s: no matching channels found.\n\n', roi_label);
        continue;
    end

    fprintf('\nROI: %s (channels: %s)\n', ...
        roi_label, strjoin(all_labels_full(roi_idx), ', '));
    fprintf(fid, 'ROI: %s (channels: %s)\n', ...
        roi_label, strjoin(all_labels_full(roi_idx), ', '));

    % average epochs over ROI channels
    R_roi = squeeze(mean(epochs_remembered(roi_idx,:,:), 1));
    K_roi = squeeze(mean(epochs_recognised(roi_idx,:,:), 1));
    M_roi = squeeze(mean(epochs_not_recognised(roi_idx,:,:), 1));

    for w = 1:nWin

        idx_w = win_idx{w};
        t_start = win_times(w,1)*1000;
        t_end   = win_times(w,2)*1000;

        % trial-wise mean amplitudes in this window
        amp_R = mean(R_roi(idx_w,:), 1);
        amp_K = mean(K_roi(idx_w,:), 1);
        amp_M = mean(M_roi(idx_w,:), 1);

        % rm ANOVA
        data = [amp_R(:), amp_K(:), amp_M(:)];
        T = array2table(data, 'VariableNames', {'R','K','M'});
        Within = table(categorical({'R','K','M'})', ...
                       'VariableNames', {'Condition'});

        rm = fitrm(T, 'R-M~1', 'WithinDesign', Within);
        ran = ranova(rm);

        F_val  = ran.F(1);
        p_val  = ran.pValue(1);
        df1    = ran.DF(1);
        df2    = ran.DF(2);

        % t tests
        [~, p_RK_here] = ttest(amp_R, amp_K);   % recollection
        [~, p_KM_here] = ttest(amp_K, amp_M);   % familiarity
        p_main(r,w) = p_val;
        p_RK(r,w)   = p_RK_here;
        p_KM(r,w)   = p_KM_here;

        erp_roi_stats.(roi_label).(win_names{w}).F      = F_val;
        erp_roi_stats.(roi_label).(win_names{w}).p_main = p_val;
        erp_roi_stats.(roi_label).(win_names{w}).p_RK   = p_RK_here;
        erp_roi_stats.(roi_label).(win_names{w}).p_KM   = p_KM_here;
        erp_roi_stats.(roi_label).(win_names{w}).df     = [df1 df2];

        fprintf(fid, '%s window (%.0f–%.0f ms): ', win_names{w}, t_start, t_end);
        fprintf(fid, 'Condition F(%d,%d)=%.2f, p=%.4f | ', df1, df2, F_val, p_val);
        fprintf(fid, 'R-K p=%.4f | K-M p=%.4f\n', p_RK_here, p_KM_here);

        fprintf('%s (%s): F(%d,%d)=%.2f, p=%.4f | R-K p=%.4f | K-M p=%.4f\n', ...
            roi_label, win_names{w}, df1, df2, F_val, p_val, p_RK_here, p_KM_here);
    end
end

fclose(fid);

save(fullfile(output_dir, sprintf('%s_ERP_ROI_stats.mat', subj_id)), ...
     'erp_roi_stats','p_main','p_RK','p_KM','win_names','win_times','roi_names_erp','-v7.3');

fprintf('\nROI-based ERP stats saved to:\n  %s\n', stats_file);

%%  FDR CORRECTION - BH

fprintf('\n FDR correction: \n');

q = 0.05;  

% main effects
vec = p_main(~isnan(p_main));
[ps, ii] = sort(vec(:));
m = numel(ps);
thr = (1:m)'/m * q;
if any(ps <= thr)
    crit_main = max(ps(ps <= thr));
else
    crit_main = 0;
end
sig_main = p_main <= crit_main;

% R-K
vec = p_RK(~isnan(p_RK));
[ps, ii] = sort(vec(:));
m = numel(ps);
thr = (1:m)'/m * q;
if any(ps <= thr)
    crit_RK = max(ps(ps <= thr));
else
    crit_RK = 0;
end
sig_RK = p_RK <= crit_RK;

% K-M
vec = p_KM(~isnan(p_KM));
[ps, ii] = sort(vec(:));
m = numel(ps);
thr = (1:m)'/m * q;
if any(ps <= thr)
    crit_KM = max(ps(ps <= thr));
else
    crit_KM = 0;
end
sig_KM = p_KM <= crit_KM;

fprintf('FDR critical values: main=%.4f, R-K=%.4f, K-M=%.4f\n', ...
        crit_main, crit_RK, crit_KM);

ERP.subject_id = subj_id;
ERP.time = time_points;
ERP.conditions = {'R','K','M'};
ERP.erp_R = erp_R;  
ERP.erp_K = erp_K;
ERP.erp_M = erp_M;
ERP.diff_RK = erp_R - erp_K;
ERP.diff_KM = erp_K - erp_M;

ERP.roi_means = erp_roi_stats;       
ERP.sig_main  = sig_main;           
ERP.sig_RK    = sig_RK;
ERP.sig_KM    = sig_KM;

save(fullfile(output_dir,'ERP_summary.mat'),'ERP','-v7.3');


%% Time–Frequency Analysis (Morlet)


fprintf('\n Time–Frequency Analysis (Morlet): \n');

% un-baseline-corrected data
epochs_remembered     = all_epochs_struct_raw.remembered;
epochs_recognised     = all_epochs_struct_raw.recognised;
epochs_not_recognised = all_epochs_struct_raw.not_recognised;
chan_labels           = {EEG_ica.chanlocs.labels};
[num_chans, num_time_points, ~] = size(epochs_remembered);

time_points = linspace(-0.4, (num_time_points/new_srate) - 0.4, num_time_points);

tf_output_dir = fullfile(output_dir, 'tf_results');
if ~exist(tf_output_dir, 'dir')
    mkdir(tf_output_dir);
    fprintf('Created TF results directory: %s\n', tf_output_dir);
end

roi_definitions = struct();
roi_definitions.MidFrontal   = {'Fz', 'FCz', 'AFz', 'F1', 'F2', 'FC1', 'FC2'};
roi_definitions.LeftParietal = {'P3', 'CP3', 'P1', 'P5', 'CP1', 'PO3'};
roi_definitions.ControlOccipital = {'O1', 'O2', 'Oz'};

roi_names = fieldnames(roi_definitions);

fprintf('\nDefined %d ROIs for TF analysis:\n', numel(roi_names));
disp(roi_names);

min_freq  = 4;
max_freq  = 20;
num_freqs = 50;
frex      = logspace(log10(min_freq), log10(max_freq), num_freqs);
n_cycles  = logspace(log10(3), log10(30), num_freqs);

% Baseline window 
basel_start_t = -0.4;
basel_end_t   = 0;
[~, basel_start_idx] = min(abs(time_points - basel_start_t));
[~, basel_end_idx]   = min(abs(time_points - basel_end_t));

conditions_epochs = {epochs_remembered, epochs_recognised, epochs_not_recognised};
condition_names   = {'Remembered', 'Recognised', 'Not Recognised'};

for r = 1:numel(roi_names)

    roi_label = roi_names{r};
    roi_chans = roi_definitions.(roi_label);
    roi_idx   = find(ismember(chan_labels, roi_chans));

    if isempty(roi_idx)
        fprintf(' Skipping ROI %s (no valid channels found)\n', roi_label);
        continue;
    end

    fprintf('\n ROI: %s (channels: %s)\n', roi_label, strjoin(chan_labels(roi_idx), ', '));

    figure('Name', sprintf('TF_%s', roi_label), 'Position', [100 100 1200 800]);

    clim_range = [-3 3];  % in dB

    for cond_idx = 1:numel(conditions_epochs)

        current_epochs = conditions_epochs{cond_idx}(roi_idx, :, :);
        [~, ~, n_trials] = size(current_epochs);

        % Averaging channels in ROI 
        avg_roi_data = squeeze(mean(current_epochs, 1));   

        tf_power = zeros(num_freqs, num_time_points, n_trials);

        % power per trial
        for t_idx = 1:n_trials
            x = avg_roi_data(:, t_idx)';   
            n_data = length(x);

            for fi = 1:num_freqs

                s = n_cycles(fi) / (2 * pi * frex(fi));
                t_wavelet = -3*s : 1/new_srate : 3*s;

                morlet_wavelet = exp(2*1i*pi*frex(fi)*t_wavelet) .* ...
                                 exp(-(t_wavelet.^2) / (2*s^2));

                n_kernel = length(morlet_wavelet);
                n_conv   = n_data + n_kernel - 1;

                conv_res = ifft( fft(x, n_conv) .* fft(morlet_wavelet, n_conv) );
                conv_res = conv_res(floor(n_kernel/2)+1 : floor(n_kernel/2)+n_data);

                tf_power(fi, :, t_idx) = abs(conv_res).^2;
            end
        end

        % dB baseline-correct
        tf_power_db = 10 * log10(mean(tf_power, 3));

        baseline_mean = mean(tf_power_db(:, basel_start_idx:basel_end_idx), 2);
        tf_power_bc   = bsxfun(@minus, tf_power_db, baseline_mean);

        % plot ERSP 
        subplot(3, 1, cond_idx);
        imagesc(time_points, frex, tf_power_bc);
        set(gca, 'YDir', 'normal');
        
        colormap(jet); 
        pmax = max(abs(tf_power_bc(:)));
        if pmax < 0.5
            pmax = 0.5;   
        end
        caxis([-pmax pmax]);

        colorbar;

        title(sprintf('%s (%s ROI)', condition_names{cond_idx}, roi_label), 'FontWeight', 'bold');
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        line([0 0], get(gca, 'YLim'), 'Color','k','LineStyle','--');
        ylim([min_freq max_freq]);
    end

    sgtitle(sprintf('ROI: %s — Morlet TF Power (−0.4→0 s baseline, 4–20 Hz)', roi_label), ...
        'FontSize', 14, 'FontWeight', 'bold');

    save_name = sprintf('TF_%s_20Hz.png', roi_label);
    saveas(gcf, fullfile(tf_output_dir, save_name));
    close(gcf);

    fprintf('Saved ROI TF plot: %s\n', fullfile(tf_output_dir, save_name));
end

fprintf('\nAll ROI-based TF plots saved to: %s\n', tf_output_dir);

%%  TF DIFFERENCE MAPS + CLUSTER PERMUTATION (R-K and K-M)

fprintf('\n TF DIFF + CLUSTER tests (R-K, K-M): \n');

tf_diff_dir = fullfile(tf_output_dir, 'tf_diff_and_cluster');
if ~exist(tf_diff_dir, 'dir'), mkdir(tf_diff_dir); end

COND_KEYS   = {'R','K','M'};                   
COND_EPOCHS = {epochs_remembered, epochs_recognised, epochs_not_recognised};

n_perm = 5000;
alpha  = 0.05;

num_freqs = length(frex);
num_time_points = size(epochs_remembered,2);

cluster_tf_results = struct();

bands = struct();
bands.Theta = [4 7];
bands.Alpha = [8 12];
bands.Beta  = [13 20];
band_names = fieldnames(bands);

for r = 1:numel(roi_names)
    roi_label = roi_names{r};
    roi_chans = roi_definitions.(roi_label);
    roi_idx   = find(ismember(chan_labels, roi_chans));

    if isempty(roi_idx)
        fprintf(' Skipping ROI %s (no valid channels)\n', roi_label);
        continue;
    end

    fprintf('\n ROI: %s (channels: %s)\n', roi_label, strjoin(chan_labels(roi_idx), ', '));

    % TF for each condition 
    TF = struct();
    for ci = 1:3
        epochs_cond = COND_EPOCHS{ci}(roi_idx,:,:);
        [~,~,nTrials] = size(epochs_cond);

        if nTrials == 0
            TF.(COND_KEYS{ci}) = zeros(num_freqs, num_time_points, 0);
            continue;
        end

        roi_avg = squeeze(mean(epochs_cond, 1)); 
        tf_pow  = zeros(num_freqs, num_time_points, nTrials);

        for tt = 1:nTrials
            x = roi_avg(:,tt)';
            n_data = length(x);
            for fi = 1:num_freqs
                s = n_cycles(fi) / (2*pi*frex(fi));
                t_wav = -3*s : 1/new_srate : 3*s;
                mw = exp(2*1i*pi*frex(fi)*t_wav) .* exp(-(t_wav.^2)/(2*s^2));
                n_kernel = length(mw);
                n_conv = n_data + n_kernel - 1;
                conv_res = ifft( fft(x,n_conv) .* fft(mw,n_conv) );
                conv_res = conv_res(floor(n_kernel/2)+1 : floor(n_kernel/2)+n_data);
                tf_pow(fi,:,tt) = abs(conv_res).^2;
            end
        end

        tf_db = 10*log10(tf_pow);
        baseline_mean = mean(tf_db(:, basel_start_idx:basel_end_idx, :),2);
        TF.(COND_KEYS{ci}) = tf_db - baseline_mean;
    end

    % contrasts R-K, K-M
    contrasts = {};
    nR = size(TF.R,3); nK = size(TF.K,3); nM = size(TF.M,3);

    nRK = min(nR,nK);
    if nRK > 0
        contrasts{end+1} = struct('name','RminusK','A',TF.R(:,:,randperm(nR,nRK)), ...
                                                  'B',TF.K(:,:,randperm(nK,nRK)), 'n',nRK);
    end

    nKM = min(nK,nM);
    if nKM > 0
        contrasts{end+1} = struct('name','KminusM','A',TF.K(:,:,randperm(nK,nKM)), ...
                                                  'B',TF.M(:,:,randperm(nM,nKM)), 'n',nKM);
    end

    if isempty(contrasts)
        continue;
    end

    for cc = 1:length(contrasts)
        C = contrasts{cc};
        name = C.name; n = C.n;
        A_cur = C.A; B_cur = C.B;


        D_2d = A_cur - B_cur;
        meanD = mean(D_2d,3); stdD = std(D_2d,0,3); stdD(stdD==0)=eps;
        t_obs = meanD ./ (stdD ./ sqrt(n));

        df = n-1;
        t_thresh = tinv(1-alpha/2, df);
        sig_mask = abs(t_obs) > t_thresh;

        CCobs = bwconncomp(sig_mask);
        cm_obs = zeros(1,CCobs.NumObjects);
        for ci = 1:CCobs.NumObjects
            cm_obs(ci) = sum(abs(t_obs(CCobs.PixelIdxList{ci})));
        end
        if isempty(cm_obs), cm_obs = 0; end
        max_obs = max(cm_obs);

      
        max_perm = zeros(n_perm,1);
        rng(12345,'twister');
        for p = 1:n_perm
            signs = (randi([0 1], n, 1) * 2 - 1);
            Dp = D_2d .* reshape(signs, [1 1 n]);
            meanDp = mean(Dp,3); stdDp = std(Dp,0,3); stdDp(stdDp==0)=eps;
            t_perm = meanDp ./ (stdDp ./ sqrt(n));
            sig_perm = abs(t_perm) > t_thresh;
            CCp = bwconncomp(sig_perm);
            if CCp.NumObjects > 0
                masses = cellfun(@(idx) sum(abs(t_perm(idx))), CCp.PixelIdxList);
                max_perm(p) = max(masses);
            end
        end

        p_val = mean(max_perm >= max_obs);

        % Plot TF diff map
        figure('Name', sprintf('TFdiff_%s_%s', roi_label, name), ...
               'Color','w','Position',[200 200 1100 700]);

        imagesc(time_points, frex, meanD);
        set(gca,'YDir','normal');
        title(sprintf('%s: %s (p=%.4f)', roi_label, name, p_val));
        xlabel('Time (s)'); ylabel('Frequency (Hz)');
        colormap(turbo); colorbar;

        clim = prctile(abs(meanD(:)),99); if clim==0, clim=1; end
        caxis([-clim clim]); hold on;

        thresh_mass = prctile(max_perm,95);
        for ci = 1:CCobs.NumObjects
            if cm_obs(ci) > thresh_mass
                mask = false(size(t_obs)); mask(CCobs.PixelIdxList{ci}) = true;
                contour(time_points, frex, mask, [1 1], 'LineColor','k','LineWidth',2);
            end
        end

        saveas(gcf, fullfile(tf_diff_dir, sprintf('TFdiff_%s_%s.png', roi_label, name)));
        close(gcf);

        fprintf('  -> %s full TF: p = %.4f\n', name, p_val);

        % band-level permutation cluster test 
        for bi = 1:numel(band_names)
            band_label = band_names{bi};
            fr = bands.(band_label);
            f_idx = find(frex >= fr(1) & frex <= fr(2));
            if isempty(f_idx), continue; end

            bandA = squeeze(mean(A_cur(f_idx,:,:),1)); % time x trials
            bandB = squeeze(mean(B_cur(f_idx,:,:),1));
            D = bandA - bandB; % time x trials

            meanD = mean(D,2);
            stdD = std(D,0,2); stdD(stdD==0)=eps;
            t_obs = meanD ./ (stdD ./ sqrt(n));
            sig_mask = abs(t_obs) > t_thresh;

            CC = bwconncomp(sig_mask);
            cm_band = zeros(1,CC.NumObjects);
            for ci = 1:CC.NumObjects
                cm_band(ci) = sum(abs(t_obs(CC.PixelIdxList{ci})));
            end
            if isempty(cm_band), max_obs = 0; else max_obs = max(cm_band); end

            max_perm = zeros(n_perm,1);
            for p = 1:n_perm
                signs = (randi([0 1],n,1) * 2 - 1);
                Dp = D .* signs';
                meanDp = mean(Dp,2);
                stdDp = std(Dp,0,2); stdDp(stdDp==0)=eps;
                t_perm = meanDp ./ (stdDp ./ sqrt(n));
                sig_perm = abs(t_perm) > t_thresh;
                CCp = bwconncomp(sig_perm);
                if CCp.NumObjects > 0
                    masses = cellfun(@(idx) sum(abs(t_perm(idx))), CCp.PixelIdxList);
                    max_perm(p) = max(masses);
                end
            end

            p_band = mean(max_perm >= max_obs);
            fprintf('      Band %s: p = %.4f\n', band_label, p_band);

            % time course per band plots
            fig = figure('Name', sprintf('%s_%s_%s',roi_label,name,band_label), ...
                'Color','w','Position',[200 200 1200 350]);

            if contains(name, 'RminusK')
                condA = 'Remembered';      colorA = [0 0.4470 0.7410];      % blue
                condB = 'Recognised';      colorB = [0.8500 0.3250 0.0980];  % orange
            elseif contains(name, 'KminusM')
                condA = 'Recognised';      colorA = [0.8500 0.3250 0.0980];  % orange
                condB = 'Not Recognised';  colorB = [0.4660 0.6740 0.1880];  % green
            end

           
            plot(time_points, mean(bandA,2), 'LineWidth',1.5, 'Color', colorA); hold on;
            plot(time_points, mean(bandB,2), 'LineWidth',1.5, 'Color', colorB);

            legend(condA, condB, 'Stimulus Onset', 'Location','best');

            title(sprintf('%s (%s band) — %s (%s-%s) p=%.4f', roi_label, band_label, name, condA, condB, p_band));

            xlabel('Time (s)'); ylabel('Power (dB)');
            line([0 0], ylim, 'Color','k','LineStyle','--', 'DisplayName','Stimulus Onset');

            thresh = prctile(max_perm,95);
            for ci = 1:CC.NumObjects
                if cm_band(ci) > thresh
                    idx = CC.PixelIdxList{ci};
                    area(time_points(idx), meanD(idx), ...
                        'FaceColor',[0.8 0.8 0], 'EdgeColor','none','FaceAlpha',0.3, ...
                        'DisplayName', sprintf('Cluster %d', ci));
                end
            end

            saveas(fig, fullfile(tf_diff_dir, sprintf('%s_%s_%s_BAND.png',roi_label,name,band_label)));
            close(fig);
        end
    end
end

save(fullfile(tf_diff_dir,'cluster_tf_results.mat'),'cluster_tf_results','-v7.3');

%% SAVING FOR GROUP ANALYSIS
tf_save = struct();
tf_save.subject_id = subj_id;     
tf_save.roi_names  = roi_names;
tf_save.time_points = time_points;
tf_save.frex = frex;
tf_save.bands = bands;

tf_save.TF = TF;  % contains TF.R, TF.K, TF.M baseline‐corrected

% Save contrast p-values
tf_save.cluster_tf_results = cluster_tf_results;

save(fullfile(tf_output_dir, sprintf('TF_subject_%s.mat', subj_id)), ...
     'tf_save', '-v7.3');

fprintf('\nSaved subject TF data for group stats: TF_subject_%s.mat\n', subj_id);



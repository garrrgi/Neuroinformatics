%% -------------------------
% Load EEG dataset
% -------------------------
eeglab
EEG = pop_loadset('eeglab_data.set', fullfile(fileparts(which('eeglab')), 'sample_data'));

data = EEG.data;       % channels x samples
chan_labels = {EEG.chanlocs.labels};
%loading data and channels 

% -------------------------
% Step 1: Filtering: Notch filter at 60 Hz
% -------------------------
wo = 60/(EEG.srate/2); bo = wo/35;
[bn,an] = iirnotch(wo, bo);
data_filt = filtfilt(bn,an,data')';
%Notch filter for line noise, q value can be changed but preferred 35

% -------------------------
% Step 2: Plot PSD before & after filtering
% -------------------------
figure;
[~,~,~,~,~] = spectopo(data, size(data,2), EEG.srate);

figure;
[~,~,~,~,~] = spectopo(data_filt, size(data_filt,2), EEG.srate);
%plots spectral plots of data before and after notch filtering and how q
%value changes changes it. 

%% -------------------------
% Step 3: Referencing: Common Average Reference
% -------------------------
avg_ref = mean(data_filt, 1);
data_car = data_filt - avg_ref;
%we take mean of entire data across channels for avg reference value and
%substract it from filtered adat to get common avrage reference for the
%data. 

%% -------------------------
% Step 4: Epoching around "rt"
% -------------------------
event_latencies = [EEG.event.latency]; 
event_types     = {EEG.event.type};
target_idx = find(strcmp(event_types,'rt'));
epoch_window = round([-0.2 0.8]*EEG.srate);  
epoch_len = diff(epoch_window)+1;
%We have latency value as start of an event and rt as when trigger was
%pressed. we use rt values to epoch the window by multiplying it with
%sampling rate and then defien epoch length as +1 as the epoching is done
%keeping in mind the delay between presentation of stimulus and reaction
%time. 

% Preallocate
epochs = nan(size(data_car,1), epoch_len, length(target_idx));
for i = 1:length(target_idx)
    center = round(event_latencies(target_idx(i)));
    idx = center+epoch_window(1):center+epoch_window(2);
    if idx(1)>0 && idx(end)<=size(data_car,2)
        epochs(:,:,i) = data_car(:,idx);
    end
end
%epoching is done on the basis of epoch window and target index on car data. 

%% -------------------------
% Step 5: Baseline correction
% -------------------------
baseline_idx = 1:round(0.2*EEG.srate);
% Compute the global variance mean across trials for artifact rejection,
% you can use it either at trial level using indexing or global level using
% global var mean
global_var_mean = mean(mean(trial_var));
baseline = mean(epochs(:,baseline_idx,global_var_mean),2);
epochs_bc = epochs - baseline;
%baseline correction substracts baseline actvity from the epoched data. 

%% -------------------------
% Step 6: Improved Artifact Rejection
% -------------------------
% (a) Reject bad trials
% Computes the variance of each channel within each trial, averages that across channels per trial, 
% and then rejects trials whose average variance is more than 3 standard deviations 
% above the global variance mean (across trials).
trial_var = squeeze(var(epochs_bc,0,2));
trial_reject = (mean(trial_var,1) > mean(mean(trial_var))+3*std(mean(trial_var)));

good_trials = ~trial_reject;
epochs_clean = epochs_bc(:,:,good_trials);

fprintf('Rejected %d/%d trials\n', sum(trial_reject), length(trial_reject));
%Artifact rejectection is done across good trials in the epochs and bad
%trials are rejected leaving us with cleaned epochs. 

% (b) Reject bad channels (flat or high variance across trials)
chan_var = squeeze(var(epochs_clean,0,[2 3]));
bad_chans = (chan_var < 1e-6) | (chan_var > mean(chan_var)+3*std(chan_var));
good_chans = find(~bad_chans);
epochs_clean = epochs_clean(good_chans,:,:);

fprintf('Rejected %d/%d channels\n', sum(bad_chans), length(chan_labels));

chan_labels_clean = chan_labels(good_chans);

%step 5: Spectrogram
% === Trial- and channel-averaged spectrogram ===
% Parameters
fs    = EEG.srate;                    % sampling rate
win   = hamming(round(0.5*fs));       % 500 ms window
nover = round(0.4*fs);                % 400 ms overlap
nfft  = 2^nextpow2(length(win));      % FFT length
nChans  =32;
nTrials =30;
% Preallocate
P_all = [];
for ch = 1:nChans
    for tr = 1:nTrials
        sig = squeeze(epochs_bc(ch,:,tr));
        [~,F,T,P] = spectrogram(sig, win, nover, nfft, fs);
        if isempty(P_all)
            P_all = zeros([size(P), nChans, nTrials]);
        end
        P_all(:,:,ch,tr) = P;
    end
end
% === Average across trials and channels ===
P_mean = mean(P_all, [3 4]);
% === Plot ===
figure;
surf(T, F, 10*log10(P_mean), 'EdgeColor', 'none');
axis tight;
view(0,90);
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Trial- and Channel-averaged Spectrogram');
colorbar;
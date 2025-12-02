load("/MATLAB Drive/sampleEEGdata (1).mat")
% Input parameters
chan2use = 'fcz';
min_freq = 3;
max_freq = 30;
num_freq = 20;

%TFR with morlet wavelet number 6 in freqs * time * trials matrix in
%power_all variable

waveletnum = 6;

% channel index
chanIdx = find(strcmpi({EEG.chanlocs.labels}, chan2use));
if isempty(chanIdx)
    error(['Channel ' chan2use ' not found in EEG.chanlocs']);
end

% data: time x trials
data = squeeze(EEG.data(chanIdx, :, :));  % (time x trials)
fs = EEG.srate;                          
timeVec = EEG.times / 1000;               % convert ms → s
cfreqs = linspace(min_freq, max_freq, num_freq); % center frequencies

load("/MATLAB Drive/power_All.mat")

%Power average 
power_avg = mean(power_all, 3);

%Time-Frequency Power Map
figure('Color','w');
imagesc(timeVec, cfreqso, power_avg);
set(gca, 'YDir', 'normal');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title(sprintf('TFR (%s) – Morlet wavelet #%d', upper(chan2use), waveletnum));
colorbar;

%Edge trimming
time_s = dsearchn(EEG.times', -500);
time_e = dsearchn(EEG.times', 1200);
eegpower = power_all(:, time_s:time_e, :);
tftimes = EEG.times(time_s:time_e);
nTimepoints = length(tftimes);

%Plot
%Parameters
voxel_pval = 0.01;
cluster_pval = 0.05;
n_permutes = 2000;

baseidx = [dsearchn(tftimes', -500), dsearchn(tftimes', -100)]; %baseline range

%Trial-level baseline normalization 
normed_power = zeros(size(eegpower));  %[freq x time x trials]
% Loop through trials
for tr = 1:size(eegpower, 3)
    trialPower = eegpower(:, :, tr);  
    basePower = mean(trialPower(:, baseidx(1):baseidx(2)), 2); 
    normed_power(:, :, tr) = 10 * log10(bsxfun(@rdivide, trialPower, basePower)); % dB normalization
end

% Average across trials
realmean = mean(normed_power, 3); 

%baseline-normalized average TFR 
figure('Color','w');
imagesc(tftimes, cfreqso, realmean);
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('Baseline-normalized TFR (%s, %.1f–%.1f Hz)', upper(chan2use), min_freq, max_freq));
colorbar;


% shuffle data i.e. destroy time-locking to the stimuli to obtain 1000
% equivalents of realmean and store in permuted_vals

%%Create null distribution by time-shuffling

n_perm = 1000;  % number of permutations
[nFreq, nTimes, nTrials] = size(normed_power);

permuted_value = zeros(nFreq, nTimes, n_perm);

for perm_i = 1:n_perm
    shuffled_trials = zeros(nFreq, nTimes, nTrials);
    
    % Shuffle each trial's time points independently
    for tr = 1:nTrials
        shift_amt = randi(nTimes);  
        shuffled_trials(:,:,tr) = circshift(normed_power(:,:,tr), [0 shift_amt]);
    end
    
    % Average across trials after shuffling
    permuted_value(:,:,perm_i) = mean(shuffled_trials, 3);
end

%Create a z-score metric
%Name it zmap

% Compute mean and std across permutations for each voxel
perm_mean = mean(permuted_value, 3);
perm_std  = std(permuted_value, 0, 3);
perm_std(perm_std == 0) = eps;

% Compute z-map: how deviant the observed map is from the permutation null
zmap = (realmean - perm_mean) ./ perm_std;

%Visualize the z-map
figure('Color','w');
imagesc(tftimes, cfreqs, zmap);
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title('Z-score map of observed power vs. shuffled null');
colorbar;

% Calculate those bins where z-score threshold exceeds p val say 0.05 and
% store results in threshmean

%%Threshold z-map based on p-value criterion
p_threshold = 0.05;             % desired significance level
two_tailed = true;              % set false for one-tailed test

if two_tailed
    z_thresh = norminv(1 - p_threshold/2);
else
    z_thresh = norminv(1 - p_threshold);
end

%thresholded map
threshmean = zeros(size(zmap));

% Apply threshold
threshmean(abs(zmap) >= z_thresh) = zmap(abs(zmap) >= z_thresh);

%%Visualize thresholded z-map
figure('Color','w');
contourf(tftimes, cfreqs, threshmean, 20, 'LineColor', 'none'); % 20 levels for smooth contours
set(gca, 'YDir', 'normal');
xlabel('Time (ms)');
ylabel('Frequency (Hz)');
title(sprintf('Thresholded Z-map (p < %.2f, |z| > %.2f)', p_threshold, z_thresh));
colorbar;
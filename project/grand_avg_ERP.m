clear; clc;

base_dir = '/home/ozoswita/Dataset/Essex_Movie';

subjects = {'sub-001','sub-002','sub-003','sub-004','sub-005', ...
            'sub-006','sub-007','sub-008','sub-009','sub-011'};

chan_names_of_interest = {'Fz', 'FCz', 'AFz', 'F1', 'F2', 'FC1', 'FC2', ...
                          'P3', 'CP3', 'P1', 'P5', 'CP1', 'PO3', 'O1', 'Oz', 'O2'};

roi_def = struct();
roi_def.MidFrontal        = {'Fz','FCz','AFz','F1','F2','FC1','FC2'};
roi_def.LeftParietal      = {'P3','CP3','P1','P5','CP1','PO3'};
roi_def.ControlOccipital  = {'O1','Oz','O2'};

roi_list = fieldnames(roi_def);

win_names = {'FN400','LPC'};
win_times = [0.15 0.45;   % FN400
             0.50 0.80];  % LPC

ROI_data = struct();
for r = 1:numel(roi_list)
    for w = 1:numel(win_names)
        ROI_data.(roi_list{r}).(win_names{w}).R = [];
        ROI_data.(roi_list{r}).(win_names{w}).K = [];
        ROI_data.(roi_list{r}).(win_names{w}).M = [];
    end
end


% ROI means per subject

for s = 1:length(subjects)
    subj = subjects{s};
    fprintf('\n %s: \n', subj);

    erp_file = fullfile(base_dir, subj, 'unit_test_results', 'ERP_summary.mat');
    if ~exist(erp_file, 'file')
        fprintf('\n Missing ERP_summary for %s, skipping.\n', subj);
        continue;
    end
    load(erp_file, 'ERP');   

    eeg_file = fullfile(base_dir, subj, 'unit_test_results', [subj '_postICA.set']);
    if ~exist(eeg_file, 'file')
        fprintf('\n Missing postICA set for %s, skipping.\n', subj);
        continue;
    end
    EEG_ica = pop_loadset('filename', [subj '_postICA.set'], ...
                          'filepath', fullfile(base_dir, subj, 'unit_test_results'));
    all_labels = {EEG_ica.chanlocs.labels};
    time = ERP.time;

     chan_names_found = {};
    for n = 1:length(chan_names_of_interest)
        if any(strcmpi(all_labels, chan_names_of_interest{n}))
            chan_names_found{end+1} = chan_names_of_interest{n};
        end
    end

    % Sanity chexck
    if length(chan_names_found) ~= size(ERP.erp_R,1)
        warning(' %s: mismatch between channel subset and ERP size (found %d names, ERP has %d chans).', ...
                 subj, length(chan_names_found), size(ERP.erp_R,1));
       
    end

    for r = 1:numel(roi_list)
        roi_name  = roi_list{r};
        roi_chans = roi_def.(roi_name);

         roi_rows = find(ismember(chan_names_found, roi_chans));

        if isempty(roi_rows)
            fprintf('%s: No channels from %s present in the ERP subset, inserting NaN\n', ...
                    subj, roi_name);
            for w = 1:numel(win_names)
                ROI_data.(roi_name).(win_names{w}).R(end+1) = NaN;
                ROI_data.(roi_name).(win_names{w}).K(end+1) = NaN;
                ROI_data.(roi_name).(win_names{w}).M(end+1) = NaN;
            end
            continue;
        end

      
        R_roi = squeeze(mean(ERP.erp_R(roi_rows,:), 1));  
        K_roi = squeeze(mean(ERP.erp_K(roi_rows,:), 1));
        M_roi = squeeze(mean(ERP.erp_M(roi_rows,:), 1));

        for w = 1:numel(win_names)
            t1 = win_times(w,1);
            t2 = win_times(w,2);
            win_idx = find(time >= t1 & time <= t2);

            mean_R = mean(R_roi(win_idx));
            mean_K = mean(K_roi(win_idx));
            mean_M = mean(M_roi(win_idx));

            ROI_data.(roi_name).(win_names{w}).R(end+1) = mean_R;
            ROI_data.(roi_name).(win_names{w}).K(end+1) = mean_K;
            ROI_data.(roi_name).(win_names{w}).M(end+1) = mean_M;
        end
    end
end

%% Group rmANOVA per ROI × window

for r = 1:numel(roi_list)
    roi_name = roi_list{r};
    for w = 1:numel(win_names)
        win_name = win_names{w};

        R = ROI_data.(roi_name).(win_name).R;
        K = ROI_data.(roi_name).(win_name).K;
        M = ROI_data.(roi_name).(win_name).M;

        % remove subjects with NaN in any condition
        valid = ~isnan(R) & ~isnan(K) & ~isnan(M);
        Rv = R(valid);
        Kv = K(valid);
        Mv = M(valid);

        if numel(Rv) < 3
            fprintf('\nROI %s | %s: Not enough valid subjects, skipping ANOVA. (n=%d)\n', ...
                    roi_name, win_name, numel(Rv));
            continue;
        end

        T = table(Rv(:), Kv(:), Mv(:), 'VariableNames', {'R','K','M'});
        Within = table(categorical({'R','K','M'})', 'VariableNames', {'Condition'});

        rm = fitrm(T, 'R-M~1', 'WithinDesign', Within);
        ran = ranova(rm);

        fprintf('\nROI: %s | %s window\n', roi_name, win_name);
        disp(ran);
    end
end


%% Group Bar plot FN400 


figure('Color','w','Position',[200 200 1200 500]);

bar_colors = [0 0.447 0.741;    % R = blue
              0.850 0.325 0.098;% K = orange
              0.466 0.674 0.188]; % M = green

for r = 1:numel(roi_list)
    
    roi_name = roi_list{r};

    R = ROI_data.(roi_name).FN400.R;
    K = ROI_data.(roi_name).FN400.K;
    M = ROI_data.(roi_name).FN400.M;

    valid = ~isnan(R) & ~isnan(K) & ~isnan(M);
    R = R(valid); K = K(valid); M = M(valid);

    means = [mean(R), mean(K), mean(M)];
    sems  = [std(R)/sqrt(numel(R)), std(K)/sqrt(numel(K)), std(M)/sqrt(numel(M))];

    subplot(1, numel(roi_list), r);

  
    b = bar(1:3, means, 'FaceColor','flat', 'LineWidth',1.5); hold on;


    b.CData = bar_colors;

    % error bars
    errorbar(1:3, means, sems, 'k', 'LineStyle','none', 'LineWidth',1.8);

    set(gca, 'XTick', 1:3, 'XTickLabel', {'R','K','M'}, ...
             'FontSize',14, 'FontWeight','bold');
    title(sprintf('%s – FN400', roi_name), 'FontSize',16,'FontWeight','bold');
    ylabel('Amplitude (µV)', 'FontSize',14,'FontWeight','bold');
    grid on;
end

saveas(gcf, fullfile(base_dir,'ERP_Group_FN400_Barplots.png'));

close(gcf);  



% Planned contrasts for Hypothesis 1

stats_file = fullfile(base_dir,'Group_Results_H1_stats.txt');
fid = fopen(stats_file,'w');

fprintf(fid,'GROUP ERP RESULTS for HYPOTHESIS 1\n');
fprintf(fid,'Testing ordering R > K > M using one-sided paired t-tests\n\n');

for r = 1:numel(roi_list)
    roi_name = roi_list{r};

    for w = 1:numel(win_names)
        win_name = win_names{w};

        R = ROI_data.(roi_name).(win_name).R;
        K = ROI_data.(roi_name).(win_name).K;
        M = ROI_data.(roi_name).(win_name).M;

        valid = ~isnan(R) & ~isnan(K) & ~isnan(M);
        Rv = R(valid); Kv = K(valid); Mv = M(valid);

        if numel(Rv) < 3
            fprintf(fid,'ROI %s | %s  - NOT ENOUGH SUBJECTS (n=%d)\n\n', ...
                    roi_name, win_name, numel(Rv));
            continue;
        end

       
        [~, p_RK_two, ~, stats_RK] = ttest(Rv, Kv);
        [~, p_KM_two, ~, stats_KM] = ttest(Kv, Mv);

        p_RK = p_RK_two / 2;  % one-sided R > K
        p_KM = p_KM_two / 2;  % one-sided K > M

        fprintf(fid,'ROI %s | %s (n=%d)\n', roi_name, win_name, numel(Rv));
        fprintf(fid,'Means: R=%.3f  K=%.3f  M=%.3f\n', mean(Rv), mean(Kv), mean(Mv));
        fprintf(fid,'R-K: t(%d)=%.3f  p_one=%.5f\n', stats_RK.df, stats_RK.tstat, p_RK);
        fprintf(fid,'K-M: t(%d)=%.3f  p_one=%.5f\n\n', stats_KM.df, stats_KM.tstat, p_KM);
    end
end
fclose(fid);
fprintf('\nStats saved to: %s\n', stats_file);

%% Group Bar plot for LPC window 


figure('Color','w','Position',[200 200 1200 450]);


% R / K / M colors
bar_colors = [0 0.447 0.741;    % blue  R
              0.850 0.325 0.098;% orange K
              0.466 0.674 0.188]; % green M

for r = 1:numel(roi_list)
    roi_name = roi_list{r};

    R = ROI_data.(roi_name).LPC.R;
    K = ROI_data.(roi_name).LPC.K;
    M = ROI_data.(roi_name).LPC.M;

    valid = ~isnan(R) & ~isnan(K) & ~isnan(M);
    R = R(valid); K = K(valid); M = M(valid);

    means = [mean(R), mean(K), mean(M)];
    sems  = [std(R)/sqrt(numel(R)), std(K)/sqrt(numel(K)), std(M)/sqrt(numel(M))];

    subplot(1, numel(roi_list), r);

    b = bar(1:3, means, 'FaceColor','flat', 'LineWidth',1.5); hold on;
    b.CData = bar_colors;

    % error bars
    errorbar(1:3, means, sems, 'k','LineStyle','none','LineWidth',1.8);

    set(gca, 'XTick',1:3, 'XTickLabel',{'R','K','M'}, ...
             'FontSize',14,'FontWeight','bold');
    title(sprintf('%s – LPC', roi_name), 'FontSize',16,'FontWeight','bold');
    ylabel('Amplitude (µV)', 'FontSize',14,'FontWeight','bold');

    grid on;
end

saveas(gcf, fullfile(base_dir,'ERP_Group_LPC_Barplots.png'));

close(gcf);

% Grand ERP across 10s
figure('Color','w','Position',[200 200 1000 600]);
hold on;
plot(time, mean(ERP.erp_R,1),'LineWidth',2);
plot(time, mean(ERP.erp_K,1),'LineWidth',2);
plot(time, mean(ERP.erp_M,1),'LineWidth',2);

xline(0,'--k');
xline(0.15,':r'); xline(0.45,':r');
xline(0.50,':b'); xline(0.80,':b');

legend({'R','K','M'});
ylabel('µV'); xlabel('Time (s)');
title('Grand ERP – MidFrontal');
grid on;

% Difference plots 
figure('Color','w');
plot(time, mean(ERP.erp_R-ERP.erp_K,1),'LineWidth',2); hold on;
plot(time, mean(ERP.erp_K-ERP.erp_M,1),'LineWidth',2);
legend({'R-K','K-M'});
xline(0,'--k');
ylabel('µV'); xlabel('Time (s)');
title('Difference Waves – MidFrontal');
grid on;


%%  GRAND AVERAGE ERP (−0.4 to 0.8s)


output_dir = fullfile(base_dir, 'group_plots');
if ~exist(output_dir, 'dir'); mkdir(output_dir); end

FN_start = 0.15; FN_end = 0.45;
LPC_start = 0.50; LPC_end = 0.80;

for r = 1:numel(roi_list)
    roi_name = roi_list{r};

    Rmat = []; Kmat = []; Mmat = [];

    for s = 1:length(subjects)
        subj = subjects{s};

        erp_file = fullfile(base_dir, subj, 'unit_test_results', 'ERP_summary.mat');
        if ~exist(erp_file,'file'), continue; end
        load(erp_file,'ERP');

        eeg_file = fullfile(base_dir, subj, 'unit_test_results', [subj '_postICA.set']);
        if ~exist(eeg_file,'file'), continue; end
        EEG_ica = pop_loadset('filename',[subj '_postICA.set'], ...
                              'filepath', fullfile(base_dir, subj, 'unit_test_results'));
        all_labels = {EEG_ica.chanlocs.labels};

        chan_names_found = {};
        for c = 1:length(chan_names_of_interest)
            if any(strcmpi(all_labels, chan_names_of_interest{c}))
                chan_names_found{end+1} = chan_names_of_interest{c};
            end
        end

        roi_chans = roi_def.(roi_name);
        [found, idx] = ismember(roi_chans, chan_names_found);
        chan_idx = idx(found);

        if isempty(chan_idx), continue; end

        Rmat(end+1,:) = mean(ERP.erp_R(chan_idx,:),1);
        Kmat(end+1,:) = mean(ERP.erp_K(chan_idx,:),1);
        Mmat(end+1,:) = mean(ERP.erp_M(chan_idx,:),1);
    end

    % Group averages
    grand_R = mean(Rmat,1);
    grand_K = mean(Kmat,1);
    grand_M = mean(Mmat,1);

    % ERP plot
    figure('Color','w','Position',[200 200 900 500]); hold on;

    % Shaded FN400 & LPC windows
    y_limits = [-6 6];
    patch([FN_start FN_end FN_end FN_start], [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
         [1 .7 .7],'FaceAlpha',0.2,'EdgeColor','none');
    patch([LPC_start LPC_end LPC_end LPC_start], [y_limits(1) y_limits(1) y_limits(2) y_limits(2)], ...
         [.7 .7 1],'FaceAlpha',0.2,'EdgeColor','none');

 
    plot(ERP.time, grand_R,'LineWidth',2);
    plot(ERP.time, grand_K,'LineWidth',2);
    plot(ERP.time, grand_M,'LineWidth',2);

    xline(0,'--k','LineWidth',1.5);

    title(sprintf('Grand ERP – %s (−0.4 to 0.8s)', roi_name),'FontWeight','bold');
    legend({'FN400','LPC','R','K','M'},'Location','best');
    ylabel('Amplitude (µV)'); xlabel('Time (s)');
    grid on;

    xlim([-0.4 0.8]);       
    ylim(y_limits);

    saveas(gcf, fullfile(output_dir, sprintf('ERP_zoom_%s_baseline.png', roi_name)));
    close(gcf);

    % Difference plots
    figure('Color','w','Position',[200 200 900 400]); hold on;

    diff_RK = mean(Rmat-Kmat,1);
    diff_KM = mean(Kmat-Mmat,1);

    plot(ERP.time, diff_RK,'LineWidth',2);
    plot(ERP.time, diff_KM,'LineWidth',2);

    xline(0,'--k','LineWidth',1.5);
    xline(FN_start,':r'); xline(FN_end,':r');
    xline(LPC_start,':b'); xline(LPC_end,':b');

    legend({'R-K','K-M'});
    title(sprintf('Difference Waves – %s (−0.2 to 0.8s)', roi_name),'FontWeight','bold');
    ylabel('Amplitude (µV)'); xlabel('Time (s)');
    grid on;

    xlim([-0.4 0.8]);
    ylim([-4 4]);

    saveas(gcf, fullfile(output_dir, sprintf('Diff_zoom_%s_baseline.png', roi_name)));
    close(gcf);
end


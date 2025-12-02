clear; clc;

base_dir = '/home/ozoswita/Dataset/Essex_Movie';
subjects = {'sub-001','sub-002','sub-003','sub-004','sub-005',...
            'sub-006','sub-007','sub-008','sub-009','sub-011'};

roi_def = struct();
roi_def.MidFrontal        = {'Fz','FCz','AFz','F1','F2','FC1','FC2'};
roi_def.LeftParietal      = {'P3','CP3','P1','P5','CP1','PO3'};
roi_def.ControlOccipital  = {'O1','Oz','O2'};

roi_names = fieldnames(roi_def);

TF_all = struct();
sub_count = 0;

for s = 1:length(subjects)
    subj = subjects{s};
    file = fullfile(base_dir, subj, 'unit_test_results','tf_results',...
                    sprintf('TF_subject_%s.mat', subj));

    if ~exist(file,'file'), continue; end

    load(file,'tf_save');
    sub_count = sub_count + 1;

    chan_labels = tf_save.chan_labels;
    frex = tf_save.frex;
    time_points = tf_save.time_points;

    for cond = {'R','K','M'}
        C = cond{1};
        TFx = tf_save.TF.(C);  

        for r = 1:numel(roi_names)
            roi = roi_names{r};
            roi_chans = roi_def.(roi);

            idx = find(ismember(chan_labels, roi_chans));
            if isempty(idx), continue; end

            roi_tf = squeeze(mean(TFx(:,:,idx),3)); 
            TF_all.(roi).(C)(:,:,sub_count) = roi_tf;
        end
    end
end

%% PLOT ROI separated TF maps

conds = {'R','K','M'};
cond_titles = {'Remembered','Known','Missed'};

cmin = -3;   
cmax =  3;   

for r = 1:numel(roi_names)
    roi = roi_names{r};

    figure('Color','w','Position',[150 20 600 950]);

    for ci = 1:length(conds)
        TFg = mean(TF_all.(roi).(conds{ci}), 3);   

        subplot(3,1,ci);
        imagesc(time_points, frex, TFg);
        set(gca,'YDir','normal');

        title(sprintf('%s – %s', roi, cond_titles{ci}), 'FontWeight','bold');
        xlabel('Time (s)'); ylabel('Freq (Hz)');

        colormap(jet);     
        caxis([cmin cmax]);   
        colorbar;

        line([0 0], get(gca,'YLim'), 'Color','k','LineStyle','--','LineWidth',1.2);
    end

    sgtitle(sprintf('Group TF – %s ROI ', roi), ...
        'FontSize',16,'FontWeight','bold');
end



%% BAND-WISE TIME SERIES (R/K/M) 
bands.Theta = [4 7];
bands.Alpha = [8 12];
bands.Beta  = [13 20];
band_names  = fieldnames(bands);

conds       = {'R','K','M'};
cond_labels = {'Remembered','Known','Missed'};
colors      = [0 0.447 0.741;    % blue  R
               0.850 0.325 0.098;% orange K
               0.466 0.674 0.188]; % green M

for r = 1:numel(roi_names)
    roi = roi_names{r};

    figure('Color','w','Position',[100 50 900 800]);
    sgtitle(sprintf('Band Power Time Course – %s ROI (dB rel. baseline)', roi), ...
            'FontWeight','bold');

    for b = 1:numel(band_names)
        band = band_names{b};
        fr   = bands.(band);

        f_idx = find(frex >= fr(1) & frex <= fr(2));

        
        R_ts = squeeze(mean(TF_all.(roi).R(f_idx,:,:),1));  
        K_ts = squeeze(mean(TF_all.(roi).K(f_idx,:,:),1));
        M_ts = squeeze(mean(TF_all.(roi).M(f_idx,:,:),1));

        nsub = size(R_ts,2);

        mean_R = mean(R_ts,2); sem_R = std(R_ts,0,2)/sqrt(nsub);
        mean_K = mean(K_ts,2); sem_K = std(K_ts,0,2)/sqrt(nsub);
        mean_M = mean(M_ts,2); sem_M = std(M_ts,0,2)/sqrt(nsub);

        subplot(numel(band_names),1,b); hold on;

        % helper for shaded SEM
        plot_with_sem = @(mu,sem,col) ...
            fill([time_points fliplr(time_points)], ...
                 [ (mu-sem)' fliplr((mu+sem)') ], ...
                 col, 'FaceAlpha',0.15,'EdgeColor','none');

        plot_with_sem(mean_R, sem_R, colors(1,:));
        plot_with_sem(mean_K, sem_K, colors(2,:));
        plot_with_sem(mean_M, sem_M, colors(3,:));

        plot(time_points, mean_R,'Color',colors(1,:),'LineWidth',1.5);
        plot(time_points, mean_K,'Color',colors(2,:),'LineWidth',1.5);
        plot(time_points, mean_M,'Color',colors(3,:),'LineWidth',1.5);

        xline(0,'--k','LineWidth',1);  

        ylabel(sprintf('%s (dB)',band));
        if b == numel(band_names)
            xlabel('Time (s)');
        end
        grid on;
        legend(cond_labels,'Location','best');
        xlim([time_points(1) time_points(end)]);
    end
end

%% CLUSTER PERMUTATION 

alpha = 0.05;            % threshold
n_perm = 5000;          
contrasts = {'RminusK','KminusM'};

colors = [0 0.447 0.741; ...     % blue = R
          0.850 0.325 0.098; ... % orange = K
          0.466 0.674 0.188];    % green = M

fprintf('\nTEMPORAL CLUSTER PERMUTATION RESULTS :\n');

for r = 1:numel(roi_names)
    roi = roi_names{r};

    for b = 1:numel(band_names)
        band = band_names{b};
        fr = bands.(band);

        % frequency selection
        f_idx = frex >= fr(1) & frex <= fr(2);

        % extract time series time x subjects
        R_ts = squeeze(mean(TF_all.(roi).R(f_idx,:,:),1));
        K_ts = squeeze(mean(TF_all.(roi).K(f_idx,:,:),1));
        M_ts = squeeze(mean(TF_all.(roi).M(f_idx,:,:),1));

        % differences
        D_RK = R_ts - K_ts;
        D_KM = K_ts - M_ts;

        diffStruct = struct('RminusK',D_RK, 'KminusM',D_KM);

        for cc = 1:numel(contrasts)
            cname = contrasts{cc};
            D = diffStruct.(cname);

            [nT, nS] = size(D);

            % Observed t-map
            meanD = mean(D,2);
            stdD = std(D,0,2); stdD(stdD==0)=eps;
            t_obs = meanD ./ (stdD ./ sqrt(nS));

            df = nS-1;
            t_thr = tinv(1-alpha/2, df);
            mask = abs(t_obs) > t_thr;

            % find clusters in time
            CC = bwconncomp(mask);
            obsMass = zeros(CC.NumObjects,1);
            for k = 1:CC.NumObjects
                obsMass(k) = sum(abs(t_obs(CC.PixelIdxList{k})));
            end
            maxObs = max([0; obsMass]);

            % Permutation null
            maxNull = zeros(n_perm,1);
            %rng(123);

            for p = 1:n_perm
                flip = (randi([0 1],1,nS)*2 - 1);
                Dp = D .* flip;

                meanDp = mean(Dp,2);
                stdDp  = std(Dp,0,2); stdDp(stdDp==0)=eps;
                t_perm = meanDp ./ (stdDp ./ sqrt(nS));

                maskp = abs(t_perm) > t_thr;
                CCp = bwconncomp(maskp);
                if CCp.NumObjects > 0
                    masses = zeros(CCp.NumObjects,1);
                    for k = 1:CCp.NumObjects
                        masses(k) = sum(abs(t_perm(CCp.PixelIdxList{k})));
                    end
                    maxNull(p) = max(masses);
                end
            end

            critMass = prctile(maxNull, 95);
            p_cluster = mean(maxNull >= maxObs);

            fprintf('%s | %s | %s -> p_cluster = %.4f\n', roi, band, cname, p_cluster);

           
            figure('Color','w','Position',[200 200 1100 350]);
            hold on;

            y = ylim;
            for k = 1:CC.NumObjects
                if obsMass(k) > critMass
                    idx = CC.PixelIdxList{k};
                    fill([time_points(idx) fliplr(time_points(idx))], ...
                         [repmat(y(1),numel(idx),1)' fliplr(repmat(y(2),numel(idx),1)')], ...
                         [1 .85 .85], 'EdgeColor','none','FaceAlpha',0.4);
                end
            end

            plot(time_points, meanD,'k','LineWidth',2);
            xline(0,'--k','LineWidth',1.5);

            title(sprintf('%s | %s | %s (p = %.4f)', roi, band, cname, p_cluster));
            xlabel('Time (s)');
            ylabel('Diff (dB)');
            grid on;
            xlim([time_points(1) time_points(end)]);

        end
    end
end



%% GROUP CLUSTER-PERMUTATION FOR EACH CONDITION

alpha_pixel = 0.05;      % pixel-level threshold
alpha_cluster = 0.05;    % cluster-level threshold
n_perm = 2000;

conditions = {'R','K','M'};

for r = 1:length(roi_names)
    roi = roi_names{r};
    fprintf('\n ROI: %s: \n', roi);

    for ci = 1:length(conditions)
        cond = conditions{ci};
        fprintf('\n Condition %s: \n', cond);

        A = TF_all.(roi).(cond);     
        [nf, nt, ns] = size(A);

        obs = mean(A,3);             

        % null distribution
        null_maps = zeros(nf, nt, n_perm);

        rng(123);
        for p = 1:n_perm
            signs = (randi([0 1],1,ns)*2 - 1);     % +1 or -1 for each subject
            Ap = A .* reshape(signs,[1 1 ns]);     % sign flip
            null_maps(:,:,p) = mean(Ap,3);
        end

        null_mean = mean(null_maps,3);
        null_std  = std(null_maps,0,3); null_std(null_std==0)=eps;

        % z-map 
        zmap = (obs - null_mean) ./ null_std;

        % thresholding
        z_thr = norminv(1 - alpha_pixel/2);
        supra_thresh = abs(zmap) > z_thr;

     
        CC = bwconncomp(supra_thresh);
        obs_cluster_mass = zeros(CC.NumObjects,1);
        for k = 1:CC.NumObjects
            obs_cluster_mass(k) = sum(abs(zmap(CC.PixelIdxList{k})));
        end

        if isempty(obs_cluster_mass)
            max_obs = 0;
        else
            max_obs = max(obs_cluster_mass);
        end

        % permutation cluster null distributions
        max_cluster_null = zeros(n_perm,1);
        for p = 1:n_perm
            smap = (null_maps(:,:,p) - null_mean) ./ null_std;   % convert to z
            mask = abs(smap) > z_thr;

            CCp = bwconncomp(mask);
            if CCp.NumObjects > 0
                masses = zeros(CCp.NumObjects,1);
                for k = 1:CCp.NumObjects
                    masses(k) = sum(abs(smap(CCp.PixelIdxList{k})));
                end
                max_cluster_null(p) = max(masses);
            else
                max_cluster_null(p) = 0;
            end
        end

        cutoff = prctile(max_cluster_null, 100*(1-alpha_cluster));
        cluster_p = mean(max_cluster_null >= max_obs);

        fprintf('Cluster p-value = %.4f (cutoff mass=%.2f)\n', cluster_p, cutoff);

        % cluster-corrected map
        z_corr = zeros(size(zmap));
        for k = 1:CC.NumObjects
            if obs_cluster_mass(k) > cutoff
                z_corr(CC.PixelIdxList{k}) = zmap(CC.PixelIdxList{k});
            end
        end

        % plots
        figure('Color','w','Position',[300 200 1200 450]);
        subplot(1,2,1);
        imagesc(time_points, frex, zmap); set(gca,'YDir','normal');
        title(sprintf('%s ROI %s (z-map uncorrected)', roi, cond));
        xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar; colormap(turbo);

        subplot(1,2,2);
        imagesc(time_points, frex, z_corr); set(gca,'YDir','normal');
        title(sprintf('%s ROI %s (cluster-corrected)', roi, cond));
        xlabel('Time (s)'); ylabel('Frequency (Hz)'); colorbar; colormap(turbo);

    end
end

%% CLUSTER-BASED PERMUTATION FOR ALL ROIs × CONDITIONS × BANDS 

alpha = 0.05;
n_perm = 2000;
conditions = {'R','K','M'};
roi_list = fieldnames(TF_all);

bands = struct();
bands.Theta = [4 7];
bands.Alpha = [8 12];
bands.Beta  = [13 20];

for r = 1:numel(roi_list)
    roi = roi_list{r};
    fprintf('\n ROI: %s: \n', roi);

    figure('Color','w','Position',[100 50 1400 900]);
    plot_idx = 1;

    for b = fieldnames(bands)'
        band_name = b{1};
        f_range = bands.(band_name);

        f_idx = frex >= f_range(1) & frex <= f_range(2);
        band_freqs = frex(f_idx);

        for ci = 1:numel(conditions)
            cond = conditions{ci};

            %band-restricted power
            A = TF_all.(roi).(cond);          
            A = A(f_idx,:,:);                 
            [nf, nt, ns] = size(A);

            % mean power in db
            realmean = mean(A,3);

            % t-map for clusters
            meanA = mean(A,3);
            stdA  = std(A,0,3); stdA(stdA==0)=eps;
            t_obs = meanA ./ (stdA ./ sqrt(ns));

            df = ns-1;
            tthr = tinv(1-alpha/2, df);

            mask_obs = abs(t_obs) > tthr;

            CC = bwconncomp(mask_obs, 8); 
            obsMass = zeros(CC.NumObjects,1);
            for k = 1:CC.NumObjects
                obsMass(k) = sum(abs(t_obs(CC.PixelIdxList{k})));
            end
            maxObs = max([0; obsMass]);

            % NULL distribution using circular time shift
            maxNull = zeros(n_perm,1);
            rng(123);

            for p = 1:n_perm
                Ap = zeros(size(A));
                for tr = 1:ns
                    shift_amount = randi(nt);
                    Ap(:,:,tr) = circshift(A(:,:,tr), [0 shift_amount]);
                end

                mAp = mean(Ap,3);
                sAp = std(Ap,0,3); sAp(sAp==0)=eps;
                t_perm = mAp ./ (sAp ./ sqrt(ns));

                mask_perm = abs(t_perm) > tthr;
                CCp = bwconncomp(mask_perm, 8);

                if CCp.NumObjects > 0
                    masses = cellfun(@(idx) sum(abs(t_perm(idx))), CCp.PixelIdxList);
                    maxNull(p) = max(masses);
                end
            end

            critMass = prctile(maxNull, 95);
            p_val = mean(maxNull >= maxObs);

            % Create cluster mask
            t_clust = zeros(size(t_obs));
            for k = 1:CC.NumObjects
                if obsMass(k) > critMass
                    t_clust(CC.PixelIdxList{k}) = 1;
                end
            end

           
            subplot(3,3,plot_idx);
            contourf(time_points, band_freqs, realmean, 40, 'LineColor','none');
            set(gca,'YDir','normal');
            colormap(turbo);
            caxis([-3 3]);  
            colorbar;

            hold on;
            contour(time_points, band_freqs, t_clust, [1 1], 'k', 'LineWidth', 2);

            line([0 0], ylim, 'Color','k','LineStyle','--');

            title(sprintf('%s | %s | %s (p=%.4f)', roi, cond, band_name, p_val), 'FontSize', 11);
            xlabel('Time (s)'); ylabel('Freq (Hz)');

            fprintf('\n%s | %s | %s  →  p_cluster = %.4f\n', roi, cond, band_name, p_val);

            plot_idx = plot_idx + 1;
        end
    end

    sgtitle(sprintf('Cluster-Corrected TF Power Maps — ROI: %s', roi), ...
        'FontSize', 16, 'FontWeight','bold');
end



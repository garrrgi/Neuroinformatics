clear; clc;
load sampleEEGdata

channel2plot = 'o2';    
timewin      = 400;     

timewinidx = round(timewin/(1000/EEG.srate));
tapers     = dpss(timewinidx,5);  

d = detrend(squeeze(EEG.data(strcmpi(channel2plot,{EEG.chanlocs.labels}),200:200+timewinidx-1,10)));

% plot EEG data snippet
figure('Name','Tapers and FFTs','NumberTitle','off')
subplot(5,2,1)
plot(d), axis tight, axis off
title(['EEG snippet from ' upper(channel2plot)])

% plot tapers
for i=1:5
    subplot(5,2,(2*(i-1))+2)
    plot(tapers(:,i)), axis tight, axis off
end

% plot taper .* data
figure('Name','Tapered Data','NumberTitle','off')
for i=1:5
    subplot(5,2,(2*(i-1))+1)
    plot(tapers(:,i).*d'), axis tight, axis off
end

% plot fft of taper .* data
f=zeros(5,timewinidx);
for i=1:5
    subplot(5,2,(2*(i-1))+2)
    f(i,:)=fft(tapers(:,i).*d');
    plot(abs(f(i,1:timewinidx/2)).^2), axis tight, axis off
end

% Average FFTs and compare to Hanning window
figure('Name','Average FFT and Hann comparison','NumberTitle','off')
subplot(521)
plot(mean(abs(f(:,1:timewinidx/2)).^2,1)), axis tight, axis off
title('Average multitaper power')

subplot(523)
hann = .5*(1-cos(2*pi*(1:timewinidx)/(timewinidx-1)));
plot(hann), axis tight, axis off
title('Hann window')

subplot(525)
plot(hann.*d), axis tight, axis off
title('Hann * EEG')

subplot(526)
ff=fft(hann.*d);
plot(mean(abs(ff(1:timewinidx/2)).^2,1)), axis tight, axis off
title('FFT of Hann-windowed data')

%%  Figure 16.2 (multitaper TF with baseline correction) 
channel2plot    = 'pz';      
frequency2plot  = 15;        
timepoint2plot  = 200;       
nw_product      = 3;         % time-bandwidth product
times2save      = -300:50:1000; 
baseline_range  = [-200 0];  % ms baseline
timewin         = 400;       

% convert time points to indices
times2saveidx = dsearchn(EEG.times',times2save'); 
timewinidx    = round(timewin/(1000/EEG.srate));

% find baseline indices within times2save
[~,baseidx(1)] = min(abs(times2save - baseline_range(1)));
[~,baseidx(2)] = min(abs(times2save - baseline_range(2)));

% define tapers
tapers = dpss(timewinidx,nw_product);

% define frequencies
f = linspace(0,EEG.srate/2,floor(timewinidx/2)+1);

% channel index
chanidx = strcmpi(channel2plot,{EEG.chanlocs.labels});

% initialize output matrix
multitaper_tf = zeros(floor(timewinidx/2)+1,length(times2save));

% loop through time bins
for ti=1:length(times2saveidx)
    taperpow = zeros(floor(timewinidx/2)+1,1);
    for tapi = 1:size(tapers,2)-1
        data = bsxfun(@times, ...
            squeeze(EEG.data(chanidx, ...
            times2saveidx(ti)-floor(timewinidx/2)+1:times2saveidx(ti)+ceil(timewinidx/2),:)), ...
            tapers(:,tapi));
        pow = fft(data,timewinidx)/timewinidx;
        pow = pow(1:floor(timewinidx/2)+1,:);
        taperpow = taperpow + mean(pow.*conj(pow),2);
    end
    multitaper_tf(:,ti) = taperpow / tapi;
end

% baseline correction 
baselinePower = mean(multitaper_tf(:,baseidx(1):baseidx(2)),2);
db_multitaper_tf = 10*log10( multitaper_tf ./ repmat(baselinePower,1,length(times2save)) );

% Plot time course at one frequency band
figure('Name','Multitaper TF Results','NumberTitle','off')
subplot(121)
[~,freq2plotidx]=min(abs(f-frequency2plot));
plot(times2save,mean(db_multitaper_tf(freq2plotidx-2:freq2plotidx+2,:),1),'LineWidth',1.5)
title([upper(channel2plot) ' : ' num2str(frequency2plot) ' Hz (baseline-corrected)'])
xlabel('Time (ms)')
ylabel('Power (dB)')
axis square
set(gca,'xlim',[times2save(1) times2save(end)])

subplot(122)
[~,time2plotidx]=min(abs(times2save-timepoint2plot));
plot(f,db_multitaper_tf(:,time2plotidx),'LineWidth',1.5)
title([upper(channel2plot) ' : ' num2str(timepoint2plot) ' ms'])
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
axis square
set(gca,'xlim',[f(1) 40])

% --- Full time-frequency map ---
figure('Name','Full TF Map','NumberTitle','off')
contourf(times2save,f,db_multitaper_tf,40,'linecolor','none')
set(gca,'clim',[-2 2])
xlabel('Time (ms)'), ylabel('Frequency (Hz)')
title(['Power via multitaper (baseline corrected) from ' upper(channel2plot)])
colorbar

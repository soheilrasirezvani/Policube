clear; clc; close all;

%% ===================== USER CONFIG ==============================
analysisMode = 'bandpass_magnified';   % 'bandpass_magnified' or 'raw_stabilized'

videoFile = "D:\dronebasedVMM\Deep-Motion-Mag-Pytorch-main\Rightview-lab-final\9.5-10.5Hz\output_frames_P02m_30_fl9.5_fh10.5_fs60.0_n4_butter.mp4";

% Save outputs to the user's Desktop in a dedicated, timestamped folder
desktopDir = fullfile(getenv('USERPROFILE'), 'Desktop');
if isempty(desktopDir) || ~isfolder(desktopDir)
    desktopDir = pwd;
end
runTag    = datestr(now, 'yyyymmdd_HHMMSS');
outputDir = fullfile(desktopDir, ['PolicubeResults_' runTag]);

nPointsPerLine     = 10;
accelPatternLabels = {'acc0','acc1','acc2','acc3'};

% PSD method: 'periodogram' (single FFT, sharper) or 'welch' (smoother)
psdMethod   = 'welch';
welchSegLen = [];          % [] = auto (8 segments, 50% overlap)
welchOverlap = 0.5;

% Manual peak picking (click on PSD)
manualPick = false;

if strcmpi(analysisMode, 'bandpass_magnified')
    lineLabels    = {'1st landing'};
    stabilize     = false;
    magBand       = [9 11];
    accelTargetHz = 10.3;
    applyBandpass = false;
    spectraXLim   = [8 19];
elseif strcmpi(analysisMode, 'raw_stabilized')
    lineLabels    = {};
    stabilize     = true;
    magBand       = [5 20];
    accelTargetHz = [];
    applyBandpass = true;
    bandpassRange = [5 20];
    spectraXLim   = [0 20];
else
    error('Unknown analysisMode: %s', analysisMode);
end

isRawMode = strcmpi(analysisMode, 'raw_stabilized');
nfft      = 16384;

% Plot styling
set(groot, 'DefaultAxesFontName', 'Helvetica');
set(groot, 'DefaultAxesFontSize', 11);
set(groot, 'DefaultLineLineWidth', 1.0);
set(groot, 'DefaultTextInterpreter', 'tex');
set(groot, 'DefaultAxesTickLabelInterpreter', 'tex');
set(groot, 'DefaultLegendInterpreter', 'tex');

if ~exist(outputDir, 'dir'); mkdir(outputDir); end
fprintf('Output directory: %s\n', outputDir);

%% ===================== READ VIDEO ================================
v  = VideoReader(videoFile);
fs = v.FrameRate;
fprintf('Video: %s\n  %dx%d @ %.2f fps, %.2f s\n', videoFile, v.Width, v.Height, fs, v.Duration);

firstFrame = readFrame(v);
firstGray  = im2gray(firstFrame);

%% ===================== DRAW STRUCTURAL LINES =====================
nLines     = numel(lineLabels);
N          = nPointsPerLine;
lineCoords = zeros(nLines, 4);
clr        = lines(max(nLines, 1));

if nLines > 0
    fig = figure('Name', 'Draw lines on first frame', 'NumberTitle', 'off', 'Color', 'w');
    imshow(firstFrame); hold on;
    for k = 1:nLines
        title(sprintf('Draw line %d/%d: %s', k, nLines, lineLabels{k}));
        h = drawline('Color', clr(k,:), 'LineWidth', 2);
        wait(h);
        lineCoords(k,:) = [h.Position(1,:) h.Position(2,:)];
    end
else
    fig = figure('Name', 'Draw rectangle patterns', 'NumberTitle', 'off', 'Color', 'w');
    imshow(firstFrame); hold on;
end

nPts     = nLines * N;
pts0     = zeros(nPts, 2);
ptLabels = strings(nPts, 1);
for k = 1:nLines
    tt = linspace(0, 1, N).';
    xs = lineCoords(k,1) + tt * (lineCoords(k,3) - lineCoords(k,1));
    ys = lineCoords(k,2) + tt * (lineCoords(k,4) - lineCoords(k,2));
    idx = (k-1)*N + (1:N);
    pts0(idx,:)   = [xs ys];
    ptLabels(idx) = matlab.lang.makeValidName(lineLabels{k}) + "_P" + (1:N).';
    plot(xs, ys, 'o', 'Color', clr(k,:), 'MarkerFaceColor', clr(k,:), 'MarkerSize', 6);
end

%% ===================== NCC TEMPLATES ============================
nccHalfSize  = 20;
nccSearchPad = 10;
nStructPts   = nLines * N;
nccTemplates = cell(nStructPts, 1);
for j = 1:nStructPts
    cx = round(pts0(j,1)); cy = round(pts0(j,2));
    r1 = max(1, cy - nccHalfSize); r2 = min(size(firstGray,1), cy + nccHalfSize);
    c1 = max(1, cx - nccHalfSize); c2 = min(size(firstGray,2), cx + nccHalfSize);
    nccTemplates{j} = firstGray(r1:r2, c1:c2);
end
fprintf('NCC templates: %d patches.\n', nStructPts);

%% ===================== DRAW ACCELEROMETER PATTERNS ===============
nOrigLines   = nLines;
patternRects = zeros(0, 4);
if ~isempty(accelPatternLabels)
    nPat       = numel(accelPatternLabels);
    patternClr = lines(nOrigLines + nPat);
    patternClr = patternClr(nOrigLines+1:end, :);
    patternRects = zeros(nPat, 4);
    fprintf('Draw %d rectangle patterns near each accelerometer.\n', nPat);
    for k = 1:nPat
        title(sprintf('Draw RECTANGLE for %s (%d/%d)', accelPatternLabels{k}, k, nPat));
        hR = drawrectangle('Color', patternClr(k,:), 'LineWidth', 1.8);
        wait(hR);
        patternRects(k,:) = hR.Position;
        roi = round(patternRects(k,:)); roi(3:4) = max(roi(3:4), 5);
        feats = detectMinEigenFeatures(firstGray, 'ROI', roi, 'MinQuality', 0.005);
        if size(feats.Location, 1) >= N
            [~, ord] = sort(feats.Metric, 'descend');
            sel = feats.Location(ord(1:N), :);
        else
            cx = patternRects(k,1) + patternRects(k,3)/2;
            cy = patternRects(k,2) + patternRects(k,4)/2;
            pad = repmat([cx cy], N - size(feats.Location,1), 1);
            sel = [feats.Location; pad];
            warning('Pattern %s: only %d corners found.', accelPatternLabels{k}, size(feats.Location,1));
        end
        pts0 = [pts0; sel];
        idxNew = (numel(ptLabels)+1) : (numel(ptLabels)+N);
        ptLabels(idxNew,1) = matlab.lang.makeValidName(accelPatternLabels{k}) + "_P" + (1:N).';
        rectangle('Position', patternRects(k,:), 'EdgeColor', patternClr(k,:), 'LineWidth', 1.5);
        plot(sel(:,1), sel(:,2), 's', 'Color', patternClr(k,:), 'MarkerFaceColor', patternClr(k,:), 'MarkerSize', 6);
        text(patternRects(k,1), patternRects(k,2)-6, accelPatternLabels{k}, ...
             'Color', patternClr(k,:), 'FontWeight','bold', 'FontSize', 10);
    end
    lineLabels = [lineLabels, accelPatternLabels];
    nLines = numel(lineLabels);
    nPts   = size(pts0, 1);
    clr    = [clr; patternClr];
end
title('Tracking points placed'); drawnow;
saveas(gcf, fullfile(outputDir, '00_setup_first_frame.png'));

%% ===================== STABILIZATION REF POINTS ==================
refPts0 = [];
if stabilize
    refMask = true(size(firstGray));
    for k = 1:nOrigLines
        [X, Y] = meshgrid(1:size(firstGray,2), 1:size(firstGray,1));
        x1 = lineCoords(k,1); y1 = lineCoords(k,2);
        x2 = lineCoords(k,3); y2 = lineCoords(k,4);
        L2 = (x2-x1)^2 + (y2-y1)^2;
        if L2 > eps
            tt = ((X-x1)*(x2-x1) + (Y-y1)*(y2-y1)) / L2;
            tt = max(0, min(1, tt));
            px = x1 + tt*(x2-x1); py = y1 + tt*(y2-y1);
            lineMask = hypot(X-px, Y-py) <= 25;
            refMask = refMask & ~lineMask;
        end
    end
    pts_stab = detectMinEigenFeatures(firstGray, 'MinQuality', 0.01);
    loc = double(pts_stab.Location);
    keep = false(size(loc,1), 1);
    for ii = 1:numel(keep)
        x = round(loc(ii,1)); y = round(loc(ii,2));
        if x >= 1 && x <= size(refMask,2) && y >= 1 && y <= size(refMask,1)
            keep(ii) = refMask(y, x);
        end
    end
    refPts0 = loc(keep,:);
    fprintf('Stabilization: %d reference corners.\n', size(refPts0,1));
    if size(refPts0,1) < 30
        warning('Few stab features (<30). Disabling.'); stabilize = false;
    end
end

%% ===================== KLT INIT (accel pts only) =================
nAccPts = nPts - nStructPts;
if nAccPts > 0
    accPts0 = pts0(nStructPts+1:end, :);
    mainTrk = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 3, 'BlockSize', [31 31]);
    initialize(mainTrk, accPts0, firstFrame);
end
if stabilize
    refTrk = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 3, 'BlockSize', [31 31]);
    initialize(refTrk, refPts0, firstFrame);
end

%% ===================== TRACKING LOOP =============================
nFramesEst = floor(v.Duration * fs) + 5;
trkX = nan(nFramesEst, nPts);
trkY = nan(nFramesEst, nPts);
trkX(1,:) = pts0(:,1).'; trkY(1,:) = pts0(:,2).';
nccCurPos = pts0(1:nStructPts, :);

i = 1; hWait = waitbar(0, 'Tracking video...');
while hasFrame(v)
    i = i + 1;
    fr = readFrame(v); frGray = im2gray(fr);

    for j = 1:nStructPts
        tmpl = nccTemplates{j}; [tH, tW] = size(tmpl);
        cx = round(nccCurPos(j,1)); cy = round(nccCurPos(j,2));
        searchHalf = nccHalfSize + nccSearchPad;
        sr1 = max(1, cy - searchHalf); sr2 = min(size(frGray,1), cy + searchHalf);
        sc1 = max(1, cx - searchHalf); sc2 = min(size(frGray,2), cx + searchHalf);
        searchRegion = frGray(sr1:sr2, sc1:sc2);
        if size(searchRegion,1) < tH || size(searchRegion,2) < tW
            trkX(i,j) = NaN; trkY(i,j) = NaN; continue;
        end
        C = normxcorr2(tmpl, searchRegion);
        [~, maxIdx] = max(C(:));
        [peakR, peakC] = ind2sub(size(C), maxIdx);
        newY = sr1 + peakR - tH + floor(tH/2);
        newX = sc1 + peakC - tW + floor(tW/2);
        trkX(i,j) = newX; trkY(i,j) = newY;
        nccCurPos(j,:) = [newX, newY];
    end

    if nAccPts > 0
        [pos, valid] = mainTrk(fr); pos(~valid,:) = NaN;
        if stabilize
            [rpos, rvalid] = refTrk(fr);
            if nnz(rvalid) >= 6
                try
                    tform = estgeotform2d(rpos(rvalid,:), refPts0(rvalid,:), 'similarity');
                    pos = transformPointsForward(tform, pos);
                catch
                end
            end
        end
        trkX(i, nStructPts+1:end) = pos(:,1).';
        trkY(i, nStructPts+1:end) = pos(:,2).';
    end

    if mod(i,20)==0, waitbar(min(i/nFramesEst,1), hWait, sprintf('Frame %d', i)); end
end
close(hWait);

nF = i; trkX = trkX(1:nF,:); trkY = trkY(1:nF,:);
t  = (0:nF-1).' / fs;
fprintf('Tracked %d frames (%.2f s).\n', nF, t(end));

%% ===================== TIME HISTORIES ============================
Yd_raw = trkY - trkY(1,:);
Yd_raw = fillmissing(Yd_raw, 'linear', 1, 'EndValues', 'nearest');
Yd_raw = detrend(Yd_raw, 1);

if applyBandpass
    nyq = fs/2;
    bp  = min(bandpassRange, [0.99*nyq 0.99*nyq]);
    bp(1) = max(bp(1), 0.05);
    [b,a] = butter(4, bp/nyq, 'bandpass');
    Yd_raw = filtfilt(b, a, Yd_raw);
end

% Keep both the per-point accel signals (for KLT diagnostic plot) and
% the averaged accel signal (one per ROI) for modal comparison.
Yd_accAll = Yd_raw(:, nStructPts+1:end);     % per-point accel (KLT) signals

nPat = nLines - nOrigLines;
Yd          = Yd_raw(:, 1:nStructPts);
pts0_avg    = pts0(1:nStructPts, :);
ptLabels_avg = ptLabels(1:nStructPts);
for kk = 1:nPat
    cols  = nStructPts + (kk-1)*N + (1:N);
    accAvg = mean(Yd_raw(:, cols), 2, 'omitnan');
    Yd = [Yd, accAvg];
    cx = patternRects(kk,1) + patternRects(kk,3)/2;
    cy = patternRects(kk,2) + patternRects(kk,4)/2;
    pts0_avg = [pts0_avg; cx cy];
    ptLabels_avg = [ptLabels_avg; string(matlab.lang.makeValidName(accelPatternLabels{kk}))];
end
pts0 = pts0_avg; ptLabels = ptLabels_avg; nPts = size(Yd, 2);

lineIdx = cell(nLines, 1);
for k = 1:nOrigLines, lineIdx{k} = (k-1)*N + (1:N); end
for kk = 1:nPat, lineIdx{nOrigLines + kk} = nStructPts + kk; end

%% ===================== PSD (Welch or Periodogram) ================
switch lower(psdMethod)
    case 'welch'
        if isempty(welchSegLen)
            segLen = max(256, floor(nF/8));
        else
            segLen = welchSegLen;
        end
        segLen  = min(segLen, nF);
        nOver   = round(segLen * welchOverlap);
        winPSD  = hann(segLen, 'periodic');
        [Pxx, f] = pwelch(Yd, winPSD, nOver, nfft, fs);
        [Pacc_pp, ~] = pwelch(Yd_accAll, winPSD, nOver, nfft, fs);
    otherwise
        winPSD  = hann(nF, 'periodic');
        [Pxx, f] = periodogram(Yd, winPSD, nfft, fs);
        [Pacc_pp, ~] = periodogram(Yd_accAll, winPSD, nfft, fs);
end

Pline = zeros(numel(f), nLines);
for k = 1:nLines
    Pline(:,k) = mean(Pxx(:, lineIdx{k}), 2, 'omitnan');
end

% Mean and std of PSDs per line (in dB) for shaded uncertainty band
PxxdB = 10*log10(max(Pxx, eps));
Pline_mean_dB = zeros(numel(f), nLines);
Pline_std_dB  = zeros(numel(f), nLines);
for k = 1:nLines
    cols = lineIdx{k};
    Pline_mean_dB(:,k) = mean(PxxdB(:, cols), 2, 'omitnan');
    Pline_std_dB(:,k)  = std(PxxdB(:, cols), 0, 2, 'omitnan');
end

if nOrigLines > 0
    Pglobal = mean(Pline(:, 1:nOrigLines), 2, 'omitnan');
else
    Pglobal = mean(Pline, 2, 'omitnan');
end

%% ===================== PEAK DETECTION (per point + per line) =====
bandIdx = f >= magBand(1) & f <= magBand(2);
f_band  = f(bandIdx);
nBins   = numel(f_band);
df      = f(2) - f(1);
minPeakDist = min(round(0.5 / df), max(1, nBins - 2));

peakFreqPerPoint = nan(nPts, 1);
for j = 1:nPts
    Pband   = Pxx(bandIdx, j);
    PbanddB = 10*log10(max(Pband, eps));
    if nBins < 3
        [~, im] = max(PbanddB);
        peakFreqPerPoint(j) = f_band(im);
    else
        [~, locs] = findpeaks(PbanddB, 'MinPeakDistance', minPeakDist, 'MinPeakProminence', 1);
        if isempty(locs)
            [~, im] = max(PbanddB); peakFreqPerPoint(j) = f_band(im);
        else
            [~, best] = max(PbanddB(locs)); peakFreqPerPoint(j) = f_band(locs(best));
        end
    end
end

% Per-line mean ± std of the per-point peaks (this is the statistic you wanted)
peakStats = table('Size', [nLines 5], ...
    'VariableTypes', {'string','double','double','double','double'}, ...
    'VariableNames', {'Line','MeanPeakHz','StdPeakHz','MinPeakHz','MaxPeakHz'});
for k = 1:nLines
    fp = peakFreqPerPoint(lineIdx{k});
    peakStats.Line(k)        = string(lineLabels{k});
    peakStats.MeanPeakHz(k)  = mean(fp, 'omitnan');
    peakStats.StdPeakHz(k)   = std(fp, 0, 'omitnan');
    peakStats.MinPeakHz(k)   = min(fp);
    peakStats.MaxPeakHz(k)   = max(fp);
end

% Per-line peak from the AVERAGED line PSD
linePeaks = nan(nLines, 1);
for k = 1:nLines
    Pband   = Pline(bandIdx, k);
    PbanddB = 10*log10(max(Pband, eps));
    if nBins < 3
        [~, im] = max(PbanddB); linePeaks(k) = f_band(im);
    else
        [~, locs] = findpeaks(PbanddB, 'MinPeakDistance', minPeakDist, 'MinPeakProminence', 2);
        if isempty(locs)
            [~, im] = max(PbanddB); linePeaks(k) = f_band(im);
        else
            [~, best] = max(PbanddB(locs)); linePeaks(k) = f_band(locs(best));
        end
    end
end
peakStats.AvgPSDPeakHz = linePeaks;

% Global PSD peak
PglobalBand = Pglobal(bandIdx);
PglobalBanddB = 10*log10(max(PglobalBand, eps));
if nBins < 3
    [~, im] = max(PglobalBanddB); globalPeakFreqs = f_band(im);
else
    [~, glocs] = findpeaks(PglobalBanddB, 'MinPeakDistance', minPeakDist, 'MinPeakProminence', 2);
    if isempty(glocs)
        [~, im] = max(PglobalBanddB); globalPeakFreqs = f_band(im);
    else
        globalPeakFreqs = f_band(glocs);
    end
end

fprintf('\n===== PEAK STATISTICS PER LINE / PATTERN =====\n');
disp(peakStats);
writetable(peakStats, fullfile(outputDir, 'peak_statistics_mean_std.csv'));

%% ===================== MANUAL PEAK PICK (optional) ================
if manualPick
    figPick = figure('Name','Manual PSD pick','Color','w');
    semilogy(f, max(Pglobal, eps), 'k-', 'LineWidth', 2.5); hold on; grid on;
    for k = 1:nLines
        semilogy(f, max(Pline(:,k), eps), '-', 'LineWidth', 1.0);
    end
    for p = 1:numel(globalPeakFreqs)
        xline(globalPeakFreqs(p), 'r--', sprintf('auto = %.2f Hz', globalPeakFreqs(p)));
    end
    if ~isempty(accelTargetHz)
        xline(accelTargetHz, 'b-', sprintf('accel = %.2f Hz', accelTargetHz), 'LineWidth', 1.5);
    end
    xlim(spectraXLim); xlabel('Frequency [Hz]'); ylabel('PSD');
    title('Click peaks, press ENTER when done');
    selectedPeaks = [];
    while true
        [xClick, ~] = ginput(1);
        if isempty(xClick), break; end
        snapMask = f >= (xClick-0.5) & f <= (xClick+0.5);
        fSnap = f(snapMask); PSnap = Pglobal(snapMask);
        [~, iSnap] = max(PSnap);
        selectedPeaks(end+1) = fSnap(iSnap);
        xline(fSnap(iSnap), 'g-', 'LineWidth', 2);
    end
    if isempty(selectedPeaks), selectedPeaks = globalPeakFreqs; end
    selectedPeaks = sort(selectedPeaks);
    close(figPick);
else
    selectedPeaks = globalPeakFreqs;
end

%% ===================== CROSS-SPECTRUM + COHERENCE ================
hasPairs = nPat >= 2;
if hasPairs
    fprintf('\nComputing cross spectrum / coherence across %d accelerometer patterns...\n', nPat);

    patSig = zeros(nF, nPat); patLab = strings(1, nPat);
    for kk = 1:nPat
        ki = nOrigLines + kk;
        patSig(:,kk) = Yd(:, lineIdx{ki});
        patLab(kk)   = string(lineLabels{ki});
    end
    patSig = detrend(patSig, 1);

    pairs  = nchoosek(1:nPat, 2);
    nPairs = size(pairs, 1);
    pairNames = strings(1, nPairs);

    % Use pwelch/cpsd/mscohere for consistent statistics
    segLenCS = max(256, floor(nF/8));
    nOverCS  = round(segLenCS * 0.5);
    winCS    = hann(segLenCS, 'periodic');
    [~, f_cs] = cpsd(patSig(:,1), patSig(:,1), winCS, nOverCS, nfft, fs);

    CPSD_complex = zeros(numel(f_cs), nPairs);
    COH          = zeros(numel(f_cs), nPairs);
    for k = 1:nPairs
        a = pairs(k,1); b = pairs(k,2);
        CPSD_complex(:,k) = cpsd(patSig(:,a), patSig(:,b), winCS, nOverCS, nfft, fs);
        COH(:,k)          = mscohere(patSig(:,a), patSig(:,b), winCS, nOverCS, nfft, fs);
        pairNames(k) = patLab(a) + " - " + patLab(b);
    end
    CPSDmag     = abs(CPSD_complex);
    CPSDphase   = rad2deg(angle(CPSD_complex));
    CPSDmag_avg = mean(CPSDmag, 2);
    COH_avg     = mean(COH, 2);

    % Time-domain cross-correlation at the dominant peak band
    fprintf('\nTime-domain cross-correlation (after band-limiting to peak ± 0.5 Hz):\n');
    peakForXC = selectedPeaks(1);
    nyq = fs/2;
    bpXC = [max(peakForXC-0.5, 0.05) min(peakForXC+0.5, 0.99*nyq)];
    [bxc, axc] = butter(4, bpXC/nyq, 'bandpass');
    patSigBP   = filtfilt(bxc, axc, patSig);

    maxLagSec = min(2, 0.5*t(end));
    maxLag    = round(maxLagSec * fs);
    xcMat     = zeros(2*maxLag+1, nPairs);
    lagSec    = (-maxLag:maxLag).' / fs;
    bestLagSec   = zeros(nPairs, 1);
    bestLagPhase = zeros(nPairs, 1);

    for k = 1:nPairs
        a = pairs(k,1); b = pairs(k,2);
        s1 = patSigBP(:,a) - mean(patSigBP(:,a));
        s2 = patSigBP(:,b) - mean(patSigBP(:,b));
        [xc, ~] = xcorr(s1, s2, maxLag, 'normalized');
        xcMat(:,k) = xc;
        [~, iMax]  = max(abs(xc));
        bestLagSec(k)   = lagSec(iMax);
        bestLagPhase(k) = wrapTo180(360 * peakForXC * bestLagSec(k));
        fprintf('  %s: lag = %+0.4f s, phase @ %.2f Hz = %+0.1f deg\n', ...
            pairNames(k), bestLagSec(k), peakForXC, bestLagPhase(k));
    end

    xcorrStats = table(pairNames(:), bestLagSec, bestLagPhase, ...
        'VariableNames', {'Pair','Lag_s','Phase_deg_at_peak'});
    writetable(xcorrStats, fullfile(outputDir, 'xcorr_phase_at_peak.csv'));
end

%% ===================== MAC (mode shape consistency) ==============
if nOrigLines >= 1
    fprintf('\nComputing operational mode shape at f = %.3f Hz on structural line...\n', selectedPeaks(1));
    [~, ipk] = min(abs(f - selectedPeaks(1)));
    nfftMode = 2^nextpow2(nF);
    Yfft = fft(Yd .* hann(nF,'periodic'), nfftMode);
    fM   = (0:nfftMode/2).' * fs / nfftMode;
    [~, ipkM] = min(abs(fM - selectedPeaks(1)));
    modeCplx = Yfft(ipkM, :).';

    modeShapes = cell(nOrigLines, 1);
    for k = 1:nOrigLines
        ms = modeCplx(lineIdx{k});
        ms = ms / max(abs(ms));
        modeShapes{k} = ms;
    end

    if nOrigLines >= 2
        MAC = zeros(nOrigLines);
        for a = 1:nOrigLines
            for b = 1:nOrigLines
                u = modeShapes{a}; w = modeShapes{b};
                MAC(a,b) = abs(u'*w)^2 / ((u'*u)*(w'*w));
            end
        end
    else
        MAC = [];
    end
end

%% ===================== SAVE TABLES ===============================
THtbl  = array2table([t Yd], 'VariableNames', [{'time_s'} cellstr(ptLabels.')]);
writetable(THtbl, fullfile(outputDir, 'time_histories_Y_px.csv'));
PSDtbl = array2table([f Pxx], 'VariableNames', [{'freq_Hz'} cellstr(ptLabels.')]);
writetable(PSDtbl, fullfile(outputDir, 'PSD_per_point.csv'));
Tpeaks = table((1:numel(selectedPeaks))', selectedPeaks(:), ...
    'VariableNames', {'ModeNumber','Frequency_Hz'});
writetable(Tpeaks, fullfile(outputDir, 'selected_peaks.csv'));

%% ===================== FIGURE STYLE HELPERS ======================
saveHQ = @(fig, name) exportgraphics(fig, fullfile(outputDir, [name '.png']), 'Resolution', 300);

%% ===================== PLOT 1: Tracked points ====================
figT = figure('Name','Tracked points','Color','w','Position',[80 80 1300 820]);
imshow(firstFrame); hold on;
for k = 1:nLines
    idx = lineIdx{k};
    if k <= nOrigLines
        plot(lineCoords(k,[1 3]), lineCoords(k,[2 4]), 'w-', 'LineWidth', 2);
        plot(pts0(idx,1), pts0(idx,2), 'wo', 'MarkerFaceColor','w', 'MarkerSize',5);
        midx = mean(lineCoords(k,[1 3])); midy = mean(lineCoords(k,[2 4]));
        text(midx, midy-15, lineLabels{k}, 'Color','w','FontWeight','bold','FontSize',10, ...
             'BackgroundColor',[0 0 0 0.6],'HorizontalAlignment','center');
    else
        pk = k - nOrigLines;
        rectangle('Position', patternRects(pk,:),'EdgeColor','r','LineWidth',2);
        plot(pts0(idx,1), pts0(idx,2), 'rs','MarkerFaceColor','r','MarkerSize',8);
        text(patternRects(pk,1), patternRects(pk,2)-10, lineLabels{k}, ...
             'Color','r','FontWeight','bold','FontSize',10,'BackgroundColor',[0 0 0 0.6]);
    end
end
title('Tracked points and accelerometer patterns');
saveHQ(figT, '01_tracked_points');

%% ===================== PLOT 2: Time histories per line ===========
% Vertically stacked panels per structural line: each point on its own
% offset stripe (waterfall style). Accelerometer panels show the 1
% averaged signal.
figTH = figure('Name','Time histories per line','Color','w', ...
    'Position',[60 60 1300 max(700, 150*nLines)]);
tl = tiledlayout(nLines, 1, 'TileSpacing','compact','Padding','compact');
title(tl, 'Y-displacement time histories', 'FontWeight','bold');

for k = 1:nLines
    nexttile; hold on; grid on; box on;
    idx = lineIdx{k};
    cols = lines(numel(idx));

    if k <= nOrigLines
        % Waterfall: each of the N points on its own offset stripe
        sig = Yd(:, idx);
        sigRng = max(std(sig, 0, 1)) * 4;
        if sigRng <= 0 || ~isfinite(sigRng), sigRng = 1; end
        for j = 1:numel(idx)
            offs = (j-1) * sigRng;
            plot(t, sig(:,j) + offs, '-', 'Color', cols(j,:), 'LineWidth', 0.9);
            text(t(1)-0.5, offs, sprintf('P%d', j), 'Color', cols(j,:), ...
                 'FontSize', 9, 'HorizontalAlignment', 'right', 'FontWeight', 'bold');
        end
        ylim([-sigRng, numel(idx)*sigRng]);
        set(gca, 'YTick', []);
        ylabel('Y-disp [px] (stacked)');
    else
        % Single averaged accelerometer-ROI signal
        plot(t, Yd(:,idx), '-', 'Color', cols(1,:), 'LineWidth', 1.2);
        ylabel('Y-disp [px]');
    end

    title(lineLabels{k}, 'FontWeight', 'normal');
    if k == nLines, xlabel('Time [s]'); end
end
saveHQ(figTH, '02_time_histories_per_line');

%% ===================== PLOT 3: KLT accel TH on single plot =======
% You explicitly asked for the per-point accelerometer time histories all
% on a single plot (waterfall stripes for easy visual comparison).
if nAccPts > 0
    figAccTH = figure('Name','KLT accel time histories','Color','w', ...
        'Position',[60 60 1300 750]);
    hold on; grid on; box on;
    cmap = parula(size(Yd_accAll,2));
    sigRng = max(std(Yd_accAll, 0, 1)) * 4;
    if sigRng <= 0 || ~isfinite(sigRng), sigRng = 1; end
    for kk = 1:nPat
        for jj = 1:N
            j = (kk-1)*N + jj;
            offs = (j-1) * sigRng;
            plot(t, Yd_accAll(:,j) + offs, '-', 'Color', cmap(j,:), 'LineWidth', 0.7);
        end
        midOff = ((kk-1)*N + N/2) * sigRng;
        text(t(end)+0.5, midOff, accelPatternLabels{kk}, ...
             'Color','k','FontWeight','bold','FontSize',11);
    end
    set(gca,'YTick',[]);
    xlabel('Time [s]');
    ylabel('Y-disp [px] (stacked per KLT point)');
    title(sprintf('All %d KLT points across %d accelerometer ROIs', nAccPts, nPat));
    saveHQ(figAccTH, '03_acc_klt_time_histories_single_plot');
end

%% ===================== PLOT 4: PSDs (dotted, bold avg, peak)  ====
% Per line: all individual PSDs dotted, line average bold black, accel
% target marked. The single key publication figure.
for k = 1:nLines
    figPSD = figure('Name',sprintf('PSD %s', lineLabels{k}),'Color','w', ...
        'Position',[80 80 1100 650]);
    ax = axes(figPSD); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    idx = lineIdx{k};

    % Individual PSDs (dotted lines)
    hInd = gobjects(numel(idx),1);
    for j = 1:numel(idx)
        hInd(j) = semilogy(ax, f, max(Pxx(:,idx(j)), eps), ':', 'LineWidth', 1.0, ...
            'Color', [0.55 0.55 0.55 0.65]);
    end

    % Mean ± 1*std band (in dB then converted back)
    if numel(idx) > 1
        mu = Pline_mean_dB(:,k);
        sd = Pline_std_dB(:,k);
        upper = 10.^((mu + sd)/10);
        lower = 10.^((mu - sd)/10);
        xfill = [f; flipud(f)];
        yfill = [max(upper,eps); flipud(max(lower,eps))];
        hBand = fill(ax, xfill, yfill, [0.2 0.2 0.2], ...
            'FaceAlpha', 0.12, 'EdgeColor', 'none');
    end

    % Bold black average
    hAvg = semilogy(ax, f, max(Pline(:,k), eps), 'k-', 'LineWidth', 3.0);

    % Auto-detected peak from the average curve
    hPkA = xline(ax, linePeaks(k), 'r--', sprintf('video peak = %.2f Hz', linePeaks(k)), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal', ...
        'LabelVerticalAlignment','top', 'Color',[0.85 0.1 0.1]);

    % Accelerometer reference (ground truth)
    if ~isempty(accelTargetHz) && isfinite(accelTargetHz)
        hPkB = xline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
            'LineWidth', 2.0, 'LabelOrientation','horizontal', ...
            'LabelVerticalAlignment','bottom', 'Color',[0.05 0.3 0.85]);
    end

    set(ax, 'YScale','log');
    xlim(ax, spectraXLim);
    xlabel(ax, 'Frequency [Hz]');
    ylabel(ax, 'PSD [px^{2}/Hz]');
    mu_pk = peakStats.MeanPeakHz(k);
    sd_pk = peakStats.StdPeakHz(k);
    title(ax, sprintf('%s   |   per-point peak: %.3f \\pm %.3f Hz  (avg-PSD peak: %.3f Hz)', ...
        lineLabels{k}, mu_pk, sd_pk, linePeaks(k)), 'FontWeight','normal');

    leg = {sprintf('Individual PSDs (n = %d)', numel(idx)), ...
           'Mean \pm 1\sigma band', ...
           'Average PSD (bold)', ...
           'Detected peak (video)'};
    if ~isempty(accelTargetHz)
        leg{end+1} = 'Accelerometer reference';
    end
    if numel(idx) > 1
        legend(ax, [hInd(1), hBand, hAvg, hPkA, hPkB], leg, 'Location','northeast');
    else
        legend(ax, [hInd(1), hAvg, hPkA, hPkB], leg([1 3 4 5]), 'Location','northeast');
    end
    saveHQ(figPSD, sprintf('04_PSD_overlay_%s', matlab.lang.makeValidName(lineLabels{k})));
end

%% ===================== PLOT 5: Global PSD overlay (all lines) ====
figG = figure('Name','Global PSD overlay','Color','w','Position',[60 60 1200 700]);
ax = axes(figG); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
cols = lines(nLines);
hLine = gobjects(nLines,1);
for k = 1:nLines
    hLine(k) = semilogy(ax, f, max(Pline(:,k), eps), ':', ...
        'LineWidth', 1.4, 'Color', cols(k,:));
end
hGlob = semilogy(ax, f, max(Pglobal, eps), 'k-', 'LineWidth', 3.0);

for p = 1:numel(globalPeakFreqs)
    xline(ax, globalPeakFreqs(p), 'r--', sprintf('peak = %.2f Hz', globalPeakFreqs(p)), ...
        'LineWidth', 1.5, 'LabelOrientation','horizontal');
end
if ~isempty(accelTargetHz) && isfinite(accelTargetHz)
    xline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
        'LineWidth', 2.0, 'Color', [0.05 0.3 0.85], 'LabelOrientation','horizontal', ...
        'LabelVerticalAlignment','bottom');
end

set(ax,'YScale','log');
xlim(ax, spectraXLim);
xlabel(ax,'Frequency [Hz]'); ylabel(ax,'PSD [px^{2}/Hz]');
title(ax,'Global PSD: per-line averages (dotted) and grand average (bold black)','FontWeight','normal');
legend(ax, [hLine; hGlob], [string(lineLabels), "GLOBAL"], 'Location','northeastoutside');
saveHQ(figG, '05_PSD_global_overlay');

%% ===================== PLOT 6: Accel KLT PSDs (single plot) ======
if nAccPts > 0
    figAP = figure('Name','Accel KLT PSDs','Color','w','Position',[60 60 1200 700]);
    ax = axes(figAP); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    cmap = parula(size(Pacc_pp,2));
    for j = 1:size(Pacc_pp,2)
        semilogy(ax, f, max(Pacc_pp(:,j), eps), ':', 'LineWidth', 0.9, ...
            'Color', cmap(j,:));
    end
    accAvgPSD = mean(Pacc_pp, 2, 'omitnan');
    semilogy(ax, f, max(accAvgPSD, eps), 'k-', 'LineWidth', 3.0);

    [~, ip] = max(accAvgPSD(bandIdx));
    accPeakHz = f_band(ip);
    xline(ax, accPeakHz, 'r--', sprintf('avg peak = %.2f Hz', accPeakHz), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal');
    if ~isempty(accelTargetHz) && isfinite(accelTargetHz)
        xline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
            'LineWidth', 2.0, 'Color',[0.05 0.3 0.85], 'LabelOrientation','horizontal', ...
            'LabelVerticalAlignment','bottom');
    end

    set(ax,'YScale','log'); xlim(ax, spectraXLim);
    xlabel(ax,'Frequency [Hz]'); ylabel(ax,'PSD [px^{2}/Hz]');
    title(ax, sprintf('All %d accelerometer-ROI KLT PSDs (dotted) + bold black average', size(Pacc_pp,2)), ...
          'FontWeight','normal');
    saveHQ(figAP, '06_acc_klt_PSDs_single_plot');
end

%% ===================== PLOT 7: Peak statistics bar chart =========
figS = figure('Name','Peak mean ± std per line','Color','w','Position',[80 80 900 500]);
ax = axes(figS); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
xc = 1:height(peakStats);
bar(ax, xc, peakStats.MeanPeakHz, 0.6, 'FaceColor',[0.4 0.6 0.85], 'EdgeColor','k');
errorbar(ax, xc, peakStats.MeanPeakHz, peakStats.StdPeakHz, 'k', ...
    'LineStyle','none','LineWidth',1.5,'CapSize',12);
if ~isempty(accelTargetHz)
    yline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
        'LineWidth',2,'Color',[0.05 0.3 0.85], 'LabelHorizontalAlignment','left');
end
set(ax,'XTick',xc,'XTickLabel',peakStats.Line);
ylabel(ax,'Detected peak frequency [Hz]');
title(ax,'Per-point peak frequency: mean \pm 1\sigma','FontWeight','normal');
saveHQ(figS, '07_peak_mean_std');

%% ===================== PLOT 8: CPSD magnitude + phase + coherence
if hasPairs
    figCS = figure('Name','Cross spectrum & coherence','Color','w','Position',[60 40 1300 900]);
    tlCS = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
    title(tlCS, sprintf('Cross-spectral analysis between %d accelerometer patterns', nPat), 'FontWeight','bold');

    % Magnitude
    nexttile; hold on; grid on; box on;
    for k = 1:nPairs
        plot(f_cs, 10*log10(max(CPSDmag(:,k), eps)), ':', 'LineWidth', 1.0, ...
            'DisplayName', pairNames(k));
    end
    hAvg = plot(f_cs, 10*log10(max(CPSDmag_avg, eps)), 'k-', 'LineWidth', 3.0, ...
        'DisplayName','Average |CPSD|');
    for p = 1:numel(selectedPeaks)
        xline(selectedPeaks(p), 'r--', sprintf('%.2f Hz', selectedPeaks(p)), ...
            'LineWidth',1.3,'LabelOrientation','horizontal','HandleVisibility','off');
    end
    if ~isempty(accelTargetHz)
        xline(accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
            'LineWidth',2,'Color',[0.05 0.3 0.85],'LabelOrientation','horizontal', ...
            'LabelVerticalAlignment','bottom','HandleVisibility','off');
    end
    ylabel('|CPSD| [dB]'); xlim(spectraXLim);
    title('Cross-spectral magnitude','FontWeight','normal');
    legend('Location','northeastoutside');

    % Phase
    nexttile; hold on; grid on; box on;
    for k = 1:nPairs
        plot(f_cs, CPSDphase(:,k), '-', 'LineWidth', 1.0, 'DisplayName', pairNames(k));
    end
    for p = 1:numel(selectedPeaks)
        xline(selectedPeaks(p), 'r--','HandleVisibility','off');
    end
    yline(0,'k--','in-phase','HandleVisibility','off');
    yline(180,'k:','out-of-phase','HandleVisibility','off');
    yline(-180,'k:','HandleVisibility','off');
    ylabel('Phase [\circ]'); ylim([-180 180]); xlim(spectraXLim);
    title('Cross-spectral phase','FontWeight','normal');
    legend('Location','northeastoutside');

    % Coherence
    nexttile; hold on; grid on; box on;
    for k = 1:nPairs
        plot(f_cs, COH(:,k), ':', 'LineWidth', 1.0, 'DisplayName', pairNames(k));
    end
    plot(f_cs, COH_avg, 'k-', 'LineWidth', 2.5, 'DisplayName','Average MSC');
    for p = 1:numel(selectedPeaks)
        xline(selectedPeaks(p), 'r--','HandleVisibility','off');
    end
    if ~isempty(accelTargetHz)
        xline(accelTargetHz, '-','Color',[0.05 0.3 0.85],'LineWidth',2,'HandleVisibility','off');
    end
    yline(0.8,'k--','HandleVisibility','off');
    xlim(spectraXLim); ylim([0 1.05]);
    xlabel('Frequency [Hz]'); ylabel('MSC \gamma^{2}');
    title('Magnitude-squared coherence','FontWeight','normal');
    legend('Location','northeastoutside');

    saveHQ(figCS, '08_cross_spectrum_coherence');

    %% ===================== PLOT 9: time-domain cross-correlation =
    figXC = figure('Name','Time-domain xcorr','Color','w','Position',[80 80 1200 600]);
    ax = axes(figXC); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    cmap = lines(nPairs);
    for k = 1:nPairs
        plot(ax, lagSec, xcMat(:,k), '-', 'LineWidth', 1.2, 'Color', cmap(k,:), ...
             'DisplayName', sprintf('%s (lag=%+0.3fs, \\phi=%+0.1f\\circ)', ...
             pairNames(k), bestLagSec(k), bestLagPhase(k)));
    end
    xline(ax, 0, 'k--', 'HandleVisibility','off');
    xlabel(ax, 'Lag [s]'); ylabel(ax, 'Normalized cross-correlation');
    title(ax, sprintf('Time-domain xcorr after band-passing around %.2f Hz', peakForXC), 'FontWeight','normal');
    legend(ax,'Location','northeastoutside');
    saveHQ(figXC, '09_time_xcorr_at_peak');
end

%% ===================== PLOT 10: mode shape & MAC =================
if nOrigLines >= 1
    figMS = figure('Name','Operational mode shape','Color','w','Position',[80 80 1200 500]);
    ax = axes(figMS); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    cmap = lines(nOrigLines);
    for k = 1:nOrigLines
        ms = modeShapes{k};
        plot(ax, 1:numel(ms), real(ms), '-o', 'Color', cmap(k,:), ...
             'LineWidth', 1.5, 'MarkerFaceColor', cmap(k,:), ...
             'DisplayName', sprintf('Re %s', lineLabels{k}));
        plot(ax, 1:numel(ms), imag(ms), '--s', 'Color', cmap(k,:), ...
             'LineWidth', 1.0, 'HandleVisibility','off');
    end
    xlabel(ax,'Point index along line'); ylabel(ax,'Normalized amplitude');
    title(ax, sprintf('Operational mode shape @ %.3f Hz (Re solid, Im dashed)', selectedPeaks(1)), ...
        'FontWeight','normal');
    legend(ax,'Location','northeastoutside');
    saveHQ(figMS, '10_mode_shape');

    if ~isempty(MAC)
        figMAC = figure('Name','MAC','Color','w','Position',[100 100 600 500]);
        imagesc(MAC); axis square; colorbar; caxis([0 1]); colormap(parula);
        set(gca,'XTick',1:nOrigLines,'XTickLabel',lineLabels(1:nOrigLines), ...
                'YTick',1:nOrigLines,'YTickLabel',lineLabels(1:nOrigLines));
        for a = 1:nOrigLines
            for b = 1:nOrigLines
                text(b, a, sprintf('%.2f', MAC(a,b)), ...
                    'HorizontalAlignment','center','Color','w','FontWeight','bold');
            end
        end
        title('MAC between lines @ peak frequency','FontWeight','normal');
        saveHQ(figMAC, '11_MAC_matrix');
    end
end

%% ===================== SUMMARY ===================================
fprintf('\n===== SUMMARY =====\n');
fprintf('Video: %s\n', videoFile);
fprintf('Mode:  %s\n', analysisMode);
fprintf('Frames: %d (%.2f s @ %.1f fps)\n', nF, t(end), fs);
fprintf('Structural lines: %d × %d pts = %d (NCC)\n', nOrigLines, N, nStructPts);
fprintf('Accelerometer ROIs: %d (KLT, %d raw points each → 1 averaged signal)\n', nPat, N);
fprintf('Analysis band: %.1f - %.1f Hz   |   PSD method: %s\n', magBand, psdMethod);
if ~isempty(accelTargetHz), fprintf('Accelerometer ground truth: %.2f Hz\n', accelTargetHz); end
fprintf('Selected peaks: '); fprintf('%.3f ', selectedPeaks); fprintf('Hz\n');
fprintf('\nPer-line peak (mean ± std):\n');
for k = 1:nLines
    fprintf('  %-20s  %.3f ± %.3f Hz   (avg-PSD peak %.3f Hz)\n', ...
        peakStats.Line(k), peakStats.MeanPeakHz(k), peakStats.StdPeakHz(k), peakStats.AvgPSDPeakHz(k));
end
if hasPairs
    fprintf('\nCross-correlation lag/phase at peak (%.3f Hz):\n', peakForXC);
    for k = 1:nPairs
        fprintf('  %-20s  lag = %+0.4f s   phase = %+0.1f deg\n', ...
            pairNames(k), bestLagSec(k), bestLagPhase(k));
    end
end
fprintf('\nResults saved to: %s\n', outputDir);

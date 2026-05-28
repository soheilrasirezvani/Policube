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
    % Each entry in lineLabels is one structural "portion" (landing,
    % flight segment, etc.) on which N points will be placed. Add or
    % remove entries to analyze however many portions you need.
    lineLabels    = {'1st landing', 'Lower flight', '2nd landing', '3rd landing'};
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

%% ===================== PEAK DETECTION (consistent rule) ==========
% ONE rule everywhere:
%   peak = bin of max(PSD_dB) within magBand, refined to sub-bin
%          precision with a 3-point parabolic fit.
% Applied identically to:
%   (a) every individual point PSD          -> peakStructPerPt / peakAccPerKLT
%   (b) every per-line averaged PSD         -> peakStats.AvgPSDPeakHz
%   (c) the global average PSD              -> globalPeakHz
% A secondary findpeaks-based estimate is computed too and stored as
% peakStats.AvgPSDPeakHz_FP so any disagreement is visible.

bandIdx = f >= magBand(1) & f <= magBand(2);
f_band  = f(bandIdx);
nBins   = numel(f_band);
df_bin  = f(2) - f(1);

% Per-KLT-point peaks for accelerometer ROIs (real std, not 0.000)
peakAccPerKLT = nan(N, nPat);
for kk = 1:nPat
    for jj = 1:N
        col = (kk-1)*N + jj;
        peakAccPerKLT(jj, kk) = paraPeakHz(Pacc_pp(:, col), f, magBand);
    end
end

% Per-point peaks on structural lines (Pxx columns 1..nStructPts)
peakStructPerPt = nan(nStructPts, 1);
for j = 1:nStructPts
    peakStructPerPt(j) = paraPeakHz(Pxx(:, j), f, magBand);
end

% Aggregate per-line peak statistics (mean ± std across point peaks)
peakStats = table('Size', [nLines 8], ...
    'VariableTypes', {'string','double','double','double','double','double','double','double'}, ...
    'VariableNames', {'Line','nPoints','MeanPeakHz','StdPeakHz','MinPeakHz','MaxPeakHz','AvgPSDPeakHz','AvgPSDPeakHz_FP'});
for k = 1:nLines
    if k <= nOrigLines
        fp = peakStructPerPt(lineIdx{k});
    else
        fp = peakAccPerKLT(:, k - nOrigLines);
    end
    peakStats.Line(k)             = string(lineLabels{k});
    peakStats.nPoints(k)          = numel(fp);
    peakStats.MeanPeakHz(k)       = mean(fp, 'omitnan');
    peakStats.StdPeakHz(k)        = std(fp, 0, 'omitnan');
    peakStats.MinPeakHz(k)        = min(fp);
    peakStats.MaxPeakHz(k)        = max(fp);
    peakStats.AvgPSDPeakHz(k)     = paraPeakHz(Pline(:,k), f, magBand);
    peakStats.AvgPSDPeakHz_FP(k)  = findpeaksPeakHz(Pline(:,k), f, magBand);
end
linePeaks = peakStats.AvgPSDPeakHz;

% Global average PSD peak (the publication number)
globalPeakHz    = paraPeakHz(Pglobal, f, magBand);
globalPeakHz_FP = findpeaksPeakHz(Pglobal, f, magBand);
globalPeakFreqs = globalPeakHz;

% Flat per-point list used by the swarm plot
peakFreqPerPoint = [peakStructPerPt; peakAccPerKLT(:)];

% Diagnostic: peak of the mean vs mean of per-point peaks
fprintf('\n===== PEAK DETECTION DIAGNOSTIC =====\n');
fprintf('  bin width df = %.4f Hz  (fs/nfft)\n', df_bin);
fprintf('  GLOBAL avg-PSD peak    : %.4f Hz  (findpeaks: %.4f Hz)\n', globalPeakHz, globalPeakHz_FP);
fprintf('  GLOBAL mean of per-pt  : %.4f +/- %.4f Hz   (n=%d)\n', ...
    mean(peakFreqPerPoint,'omitnan'), std(peakFreqPerPoint,0,'omitnan'), numel(peakFreqPerPoint));
fprintf('  (If these disagree it is physics, not algorithm:\n');
fprintf('   coherent peaks add in the average PSD, incoherent peaks average down.)\n');

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

    % Use per-KLT-point columns for accelerometer ROIs (not the
    % single averaged signal) so the overlay shows the 10 real curves.
    if k <= nOrigLines
        Pcols   = Pxx(:, lineIdx{k});
        ptPeaks = peakStructPerPt(lineIdx{k});
    else
        kk      = k - nOrigLines;
        ptCols  = (kk-1)*N + (1:N);
        Pcols   = Pacc_pp(:, ptCols);
        ptPeaks = peakAccPerKLT(:, kk);
    end
    nCurves = size(Pcols, 2);

    % Individual PSDs (dotted lines)
    hInd = gobjects(nCurves,1);
    for j = 1:nCurves
        hInd(j) = semilogy(ax, f, max(Pcols(:,j), eps), ':', 'LineWidth', 1.0, ...
            'Color', [0.55 0.55 0.55 0.65]);
    end

    % Mean ± 1*std band (in dB then converted back) - re-compute from Pcols
    if nCurves > 1
        PcolsdB  = 10*log10(max(Pcols, eps));
        mu = mean(PcolsdB, 2);
        sd = std(PcolsdB, 0, 2);
        upper = 10.^((mu + sd)/10);
        lower = 10.^((mu - sd)/10);
        xfill = [f; flipud(f)];
        yfill = [max(upper,eps); flipud(max(lower,eps))];
        hBand = fill(ax, xfill, yfill, [0.2 0.2 0.2], ...
            'FaceAlpha', 0.12, 'EdgeColor', 'none');
    end

    % Bold black average
    Pavg = mean(Pcols, 2, 'omitnan');
    hAvg = semilogy(ax, f, max(Pavg, eps), 'k-', 'LineWidth', 3.0);

    % Per-point detected peaks as ticks on the bold average curve - so the
    % user can SEE where each point's peak landed and judge the spread.
    yAt = nan(nCurves, 1);
    for j = 1:nCurves
        if ~isnan(ptPeaks(j))
            yAt(j) = interp1(f, max(Pavg, eps), ptPeaks(j), 'linear', 'extrap');
        end
    end
    hPts = scatter(ax, ptPeaks, yAt, 55, [0.85 0.1 0.1], 'v', 'filled', ...
        'MarkerEdgeColor','k', 'LineWidth', 0.5);

    % Auto-detected peak from the averaged curve
    peakAvgPSD = paraPeakHz(Pavg, f, magBand);
    hPkA = xline(ax, peakAvgPSD, 'r--', sprintf('avg-PSD peak = %.3f Hz', peakAvgPSD), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal', ...
        'LabelVerticalAlignment','top', 'Color',[0.85 0.1 0.1]);

    % Mean of per-point peaks (different statistic - shown so the gap
    % between "peak of mean" and "mean of peaks" is visible)
    muPt = mean(ptPeaks, 'omitnan');
    hPkM = xline(ax, muPt, ':', sprintf('mean of pt-peaks = %.3f Hz', muPt), ...
        'LineWidth', 1.4, 'Color', [0.5 0 0.6], ...
        'LabelOrientation','horizontal', 'LabelVerticalAlignment','middle');

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
    title(ax, sprintf('%s   |   mean of per-pt peaks: %.3f \\pm %.3f Hz   |   peak of avg PSD: %.3f Hz', ...
        lineLabels{k}, mu_pk, sd_pk, peakAvgPSD), 'FontWeight','normal');

    leg = {sprintf('Individual PSDs (n = %d)', nCurves), ...
           'Mean \pm 1\sigma band', ...
           'Average PSD (bold)', ...
           'Per-point detected peaks', ...
           'Peak of averaged PSD', ...
           'Mean of per-point peaks'};
    handles = [hInd(1), hAvg, hPts, hPkA, hPkM];
    legLabs = leg([1 3 4 5 6]);
    if nCurves > 1
        handles = [hInd(1), hBand, hAvg, hPts, hPkA, hPkM];
        legLabs = leg;
    end
    if ~isempty(accelTargetHz)
        handles(end+1) = hPkB;
        legLabs{end+1} = 'Accelerometer reference';
    end
    legend(ax, handles, legLabs, 'Location','northeast');
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

%% ===================== PLOT 5b: Per-portion comparison ===========
% One PSD per STRUCTURAL portion on a single panel, plus their
% cross-portion average in bold black. This is the natural figure
% when comparing several portions (e.g., 1st landing / Lower flight /
% 2nd landing / 3rd landing) at the proposed modal frequency.
if nOrigLines >= 2
    figPP = figure('Name','Per-portion comparison','Color','w','Position',[60 60 1200 720]);
    ax = axes(figPP); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
    portionCols = lines(nOrigLines);
    hPort = gobjects(nOrigLines, 1);
    portPeakHz = nan(nOrigLines, 1);
    for k = 1:nOrigLines
        hPort(k) = semilogy(ax, f, max(Pline(:,k), eps), '-', ...
            'LineWidth', 1.6, 'Color', portionCols(k,:));
        portPeakHz(k) = paraPeakHz(Pline(:,k), f, magBand);
    end
    Pport_avg = mean(Pline(:, 1:nOrigLines), 2, 'omitnan');
    hPortAvg = semilogy(ax, f, max(Pport_avg, eps), 'k-', 'LineWidth', 3.2);
    portAvgPeakHz = paraPeakHz(Pport_avg, f, magBand);

    % Mark each portion's peak as a small triangle on its own curve
    for k = 1:nOrigLines
        yMark = interp1(f, max(Pline(:,k), eps), portPeakHz(k), 'linear', 'extrap');
        scatter(ax, portPeakHz(k), yMark, 70, portionCols(k,:), 'v', 'filled', ...
            'MarkerEdgeColor','k', 'LineWidth', 0.5, 'HandleVisibility','off');
    end

    xline(ax, portAvgPeakHz, 'r--', sprintf('cross-portion avg-PSD peak = %.3f Hz', portAvgPeakHz), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal');
    if ~isempty(accelTargetHz) && isfinite(accelTargetHz)
        xline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
            'LineWidth', 2.0, 'Color',[0.05 0.3 0.85], 'LabelOrientation','horizontal', ...
            'LabelVerticalAlignment','bottom');
    end

    set(ax,'YScale','log'); xlim(ax, spectraXLim);
    xlabel(ax,'Frequency [Hz]'); ylabel(ax,'PSD [px^{2}/Hz]');
    title(ax, sprintf('Per-portion comparison (%d structural portions) - solid colors, cross-portion average bold black', nOrigLines), ...
        'FontWeight','normal');

    legLabs = strings(nOrigLines, 1);
    for k = 1:nOrigLines
        legLabs(k) = sprintf('%s  (peak %.3f Hz)', lineLabels{k}, portPeakHz(k));
    end
    legend(ax, [hPort; hPortAvg], [legLabs; sprintf("Cross-portion average  (peak %.3f Hz)", portAvgPeakHz)], ...
        'Location','northeastoutside');
    saveHQ(figPP, '05b_per_portion_comparison');

    % CSV with per-portion peak frequencies
    portionTbl = table(string(lineLabels(1:nOrigLines))', portPeakHz, ...
        'VariableNames', {'Portion','PeakHz'});
    portionTbl = [portionTbl; {string("CROSS_PORTION_AVG"), portAvgPeakHz}];
    writetable(portionTbl, fullfile(outputDir, 'per_portion_peaks.csv'));

    fprintf('\n===== PER-PORTION PEAK FREQUENCIES =====\n');
    disp(portionTbl);
end

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

%% ===================== PLOT 7: Overall mean +/- std pooled =======
% Pool every individual point peak (all structural-line points + all
% per-KLT-point accel peaks) and report ONE overall mean and std.
% Dots remain grouped on the x-axis for traceability.
allPeaks       = peakFreqPerPoint(~isnan(peakFreqPerPoint));
overallMean    = mean(allPeaks);
overallStd     = std(allPeaks, 0);
overallN       = numel(allPeaks);
overallSEM     = overallStd / sqrt(overallN);
if ~isempty(accelTargetHz)
    overallErrPct = 100 * abs(overallMean - accelTargetHz) / accelTargetHz;
else
    overallErrPct = NaN;
end

fprintf('\n===== OVERALL POOLED PEAK STATISTIC =====\n');
fprintf('  n = %d points (%d line + %d accel KLT)\n', overallN, nStructPts, nAccPts);
fprintf('  Overall mean +/- std = %.4f +/- %.4f Hz   (SEM = %.4f)\n', overallMean, overallStd, overallSEM);
if ~isempty(accelTargetHz)
    fprintf('  Error vs accelerometer (%.2f Hz) = %.3f%%\n', accelTargetHz, overallErrPct);
end

figS = figure('Name','Overall pooled peak mean +/- std','Color','w','Position',[80 80 1100 600]);
ax = axes(figS); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
xc = 1:height(peakStats);

% Background swarm of every individual point peak (grouped on x for context)
jitterAmp = 0.12;
hSc = [];
for k = 1:nLines
    if k <= nOrigLines
        fp = peakStructPerPt(lineIdx{k});
    else
        fp = peakAccPerKLT(:, k - nOrigLines);
    end
    xj = xc(k) + (rand(numel(fp),1) - 0.5) * 2 * jitterAmp;
    hh = scatter(ax, xj, fp, 36, [0.30 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.65, 'MarkerEdgeColor','k', 'LineWidth', 0.3, ...
        'HandleVisibility','off');
    if isempty(hSc), hSc = hh; end
end
% Single legend entry for all individual point peaks
hSc.HandleVisibility = 'on';
hSc.DisplayName = sprintf('Individual point peaks (n = %d)', overallN);

% Overall mean +/- 1 sigma band stretching across the whole plot
xRange = [0.5, height(peakStats) + 0.5];
hBand = fill(ax, [xRange fliplr(xRange)], ...
    [overallMean-overallStd overallMean-overallStd overallMean+overallStd overallMean+overallStd], ...
    [0.10 0.30 0.75], 'FaceAlpha', 0.18, 'EdgeColor','none', ...
    'DisplayName', sprintf('Overall mean \\pm 1\\sigma (n = %d)', overallN));
hMean = yline(ax, overallMean, '-', ...
    'LineWidth', 2.5, 'Color', [0.10 0.30 0.75], ...
    'DisplayName', sprintf('Overall mean = %.4f Hz', overallMean));

% Accelerometer reference + shaded ±0.1 Hz tolerance
if ~isempty(accelTargetHz) && isfinite(accelTargetHz)
    tolHz = 0.1;
    hTol = fill(ax, [xRange fliplr(xRange)], ...
        [accelTargetHz-tolHz accelTargetHz-tolHz accelTargetHz+tolHz accelTargetHz+tolHz], ...
        [0.85 0.30 0.10], 'FaceAlpha', 0.08, 'EdgeColor','none', ...
        'DisplayName', sprintf('Accel \\pm %.1f Hz', tolHz));
    hAcc = yline(ax, accelTargetHz, '--', ...
        'LineWidth', 2.0, 'Color', [0.75 0.10 0.10], ...
        'DisplayName', sprintf('Accel = %.2f Hz', accelTargetHz));
end

% One big numeric label for the overall statistic
if ~isempty(accelTargetHz)
    lblTxt = sprintf('OVERALL: %.4f \\pm %.4f Hz   (n = %d)   error vs accel: %.2f%%', ...
        overallMean, overallStd, overallN, overallErrPct);
else
    lblTxt = sprintf('OVERALL: %.4f \\pm %.4f Hz   (n = %d)', overallMean, overallStd, overallN);
end

% Y range tight around data + accel target
allVals = [peakFreqPerPoint(:); accelTargetHz(:); overallMean-overallStd; overallMean+overallStd];
yLo = min(allVals,[],'omitnan');
yHi = max(allVals,[],'omitnan');
yPad = max(0.15, 0.4*(yHi - yLo));
ylim(ax, [yLo - yPad, yHi + yPad]);
xlim(ax, xRange);

set(ax, 'XTick', xc, 'XTickLabel', peakStats.Line);
xlabel(ax, 'Tracking group (structural line / accelerometer pattern)');
ylabel(ax, 'Detected peak frequency [Hz]');
title(ax, lblTxt, 'FontWeight','bold');
legend(ax, 'Location','southoutside','Orientation','horizontal','NumColumns',2);
saveHQ(figS, '07_peak_mean_std');

% Append OVERALL row to the peak statistics CSV
overallRow = table("OVERALL", overallN, overallMean, overallStd, ...
    min(allPeaks), max(allPeaks), paraPeakHz(Pglobal, f, magBand), findpeaksPeakHz(Pglobal, f, magBand), ...
    'VariableNames', peakStats.Properties.VariableNames);
peakStatsAll = [peakStats; overallRow];
writetable(peakStatsAll, fullfile(outputDir, 'peak_statistics_mean_std.csv'));

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
        figMAC = figure('Name','MAC','Color','w','Position',[100 100 700 600]);
        imagesc(MAC); axis square; colorbar; caxis([0 1]); colormap(parula);
        set(gca,'XTick',1:nOrigLines,'XTickLabel',lineLabels(1:nOrigLines), ...
                'YTick',1:nOrigLines,'YTickLabel',lineLabels(1:nOrigLines), ...
                'XTickLabelRotation', 30);
        for a = 1:nOrigLines
            for b = 1:nOrigLines
                txtCol = 'w';
                if MAC(a,b) > 0.6, txtCol = 'k'; end
                text(b, a, sprintf('%.2f', MAC(a,b)), ...
                    'HorizontalAlignment','center','Color', txtCol, ...
                    'FontWeight','bold','FontSize',11);
            end
        end
        title(sprintf('MAC between portions @ %.3f Hz', selectedPeaks(1)),'FontWeight','normal');
        saveHQ(figMAC, '11_MAC_matrix');

        % Also save the MAC matrix to CSV for the paper appendix
        macTbl = array2table(MAC, ...
            'VariableNames', matlab.lang.makeValidName(lineLabels(1:nOrigLines)), ...
            'RowNames', lineLabels(1:nOrigLines));
        writetable(macTbl, fullfile(outputDir, 'MAC_matrix.csv'), 'WriteRowNames', true);
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
fprintf('  %-20s  %.4f ± %.4f Hz   (n=%d)\n', ...
    'OVERALL (pooled)', overallMean, overallStd, overallN);
if ~isempty(accelTargetHz)
    fprintf('  Error vs accelerometer (%.2f Hz): %.3f%%\n', accelTargetHz, overallErrPct);
end
if hasPairs
    fprintf('\nCross-correlation lag/phase at peak (%.3f Hz):\n', peakForXC);
    for k = 1:nPairs
        fprintf('  %-20s  lag = %+0.4f s   phase = %+0.1f deg\n', ...
            pairNames(k), bestLagSec(k), bestLagPhase(k));
    end
end
fprintf('\nResults saved to: %s\n', outputDir);

%% ===================== LOCAL FUNCTIONS ===========================
function fpk = paraPeakHz(P, f, band)
    % Peak frequency = argmax of dB-PSD within [band(1),band(2)] refined
    % to sub-bin resolution with a 3-point parabolic fit.
    bandIdx = f >= band(1) & f <= band(2);
    fb = f(bandIdx);
    Pb = P(bandIdx);
    if isempty(Pb), fpk = NaN; return; end
    PbdB = 10*log10(max(Pb, eps));
    [~, im] = max(PbdB);
    if im > 1 && im < numel(PbdB)
        y0 = PbdB(im-1); y1 = PbdB(im); y2 = PbdB(im+1);
        denom = (y0 - 2*y1 + y2);
        if abs(denom) > 1e-12
            delta = 0.5 * (y0 - y2) / denom;       % bin offset in [-0.5, 0.5]
            delta = max(min(delta, 0.5), -0.5);
        else
            delta = 0;
        end
        dfb = fb(2) - fb(1);
        fpk = fb(im) + delta * dfb;
    else
        fpk = fb(im);
    end
end

function fpk = findpeaksPeakHz(P, f, band)
    % Cross-check estimator: tallest findpeaks() peak inside the band.
    bandIdx = f >= band(1) & f <= band(2);
    fb = f(bandIdx);
    Pb = P(bandIdx);
    if isempty(Pb), fpk = NaN; return; end
    PbdB = 10*log10(max(Pb, eps));
    if numel(PbdB) < 3
        [~, im] = max(PbdB); fpk = fb(im); return;
    end
    [~, locs] = findpeaks(PbdB, 'MinPeakProminence', 1);
    if isempty(locs)
        [~, im] = max(PbdB); fpk = fb(im);
    else
        [~, best] = max(PbdB(locs)); fpk = fb(locs(best));
    end
end

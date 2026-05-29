% =========================================================================
% Video-based modal analysis for the Policube experiment
% -------------------------------------------------------------------------
% Pipeline (read top-to-bottom):
%   1. Load video, draw structural lines and accelerometer ROIs on frame 1
%   2. Track points: NCC for line points, KLT for accelerometer ROIs
%   3. Extract Y-displacement time histories in pixels
%   4. Crop to the active time window and compute the periodogram PSD
%      (no zero padding, df = fs / N_window)
%   5. Bin-snap peak detection in the search band
%   6. Cross-spectrum (magnitude + phase) and time-domain cross-correlation
%      between accelerometer ROIs
%   7. Save figures + CSV tables to Desktop\PolicubeResults_<timestamp>
% =========================================================================

clear; clc; close all;

%% ----- USER CONFIG -------------------------------------------------------
videoFile = "D:\dronebasedVMM\Deep-Motion-Mag-Pytorch-main\Rightview-lab-final\9.5-10.5Hz\output_frames_P02m_30_fl9.5_fh10.5_fs60.0_n4_butter.mp4";

% Each entry = one structural portion (mouse-drawn line on frame 1)
lineLabels         = {'1st landing','Lower flight','2nd landing','3rd landing'};
% Each entry = one accelerometer ROI (mouse-drawn rectangle on frame 1)
accelPatternLabels = {'acc0','acc1','acc2','acc3'};
nPointsPerLine     = 10;     % points distributed along each line / per ROI

% Modal analysis settings
analysisWindow = [1 4];      % seconds, the segment where the structure rings
magBand        = [9 11];     % Hz, peak search band (matches the magnification)
accelTargetHz  = 10.3;       % Hz, accelerometer ground-truth frequency
spectraXLim    = [8 19];     % Hz, x-axis for spectra plots

% Output folder on the Desktop (timestamped so runs don't overwrite)
desktopDir = fullfile(getenv('USERPROFILE'), 'Desktop');
if isempty(desktopDir) || ~isfolder(desktopDir), desktopDir = pwd; end
outputDir  = fullfile(desktopDir, ['PolicubeResults_' datestr(now,'yyyymmdd_HHMMSS')]);
if ~exist(outputDir,'dir'), mkdir(outputDir); end
fprintf('Output directory: %s\n', outputDir);

%% ----- READ VIDEO --------------------------------------------------------
v  = VideoReader(videoFile);
fs = v.FrameRate;
firstFrame = readFrame(v);
firstGray  = im2gray(firstFrame);
fprintf('Video: %dx%d @ %.2f fps, %.2f s\n', v.Width, v.Height, fs, v.Duration);

%% ----- DRAW STRUCTURAL LINES (mouse) ------------------------------------
nLines  = numel(lineLabels);
nPat    = numel(accelPatternLabels);
N       = nPointsPerLine;
clrAll  = lines(nLines + nPat);

figure('Name','Setup','Color','w','NumberTitle','off');
imshow(firstFrame); hold on;

lineCoords = zeros(nLines, 4);   % [x1 y1 x2 y2]
for k = 1:nLines
    title(sprintf('Draw line %d/%d: %s', k, nLines, lineLabels{k}));
    h = drawline('Color', clrAll(k,:), 'LineWidth', 2);
    wait(h);
    lineCoords(k,:) = [h.Position(1,:) h.Position(2,:)];
end

% Distribute N points along each line + grab an NCC template per point
nStructPts = nLines * N;
pts0       = zeros(nStructPts, 2);
templates  = cell(nStructPts, 1);
halfPatch  = 20;        % template half-size (px) -> 41x41 patch
for k = 1:nLines
    s    = linspace(0, 1, N).';
    xs   = lineCoords(k,1) + s * (lineCoords(k,3) - lineCoords(k,1));
    ys   = lineCoords(k,2) + s * (lineCoords(k,4) - lineCoords(k,2));
    idx  = (k-1)*N + (1:N);
    pts0(idx,:) = [xs ys];
    plot(xs, ys, 'o', 'Color', clrAll(k,:), 'MarkerFaceColor', clrAll(k,:));
    for j = idx
        cx = round(pts0(j,1));  cy = round(pts0(j,2));
        r1 = max(1, cy-halfPatch);  r2 = min(size(firstGray,1), cy+halfPatch);
        c1 = max(1, cx-halfPatch);  c2 = min(size(firstGray,2), cx+halfPatch);
        templates{j} = firstGray(r1:r2, c1:c2);
    end
end

%% ----- DRAW ACCELEROMETER ROIs (mouse) ----------------------------------
patternRects = zeros(nPat, 4);
accelPts0    = zeros(0, 2);
for k = 1:nPat
    title(sprintf('Draw rectangle %d/%d: %s', k, nPat, accelPatternLabels{k}));
    hR = drawrectangle('Color', clrAll(nLines+k,:), 'LineWidth', 1.8);
    wait(hR);
    patternRects(k,:) = hR.Position;
    roi = round(patternRects(k,:));
    feats = detectMinEigenFeatures(firstGray, 'ROI', roi, 'MinQuality', 0.005);
    if size(feats.Location,1) >= N
        [~, ord] = sort(feats.Metric, 'descend');
        sel = feats.Location(ord(1:N), :);
    else
        cx = roi(1) + roi(3)/2;  cy = roi(2) + roi(4)/2;
        sel = [feats.Location; repmat([cx cy], N-size(feats.Location,1), 1)];
        warning('ROI %s: only %d corners found, padded with ROI center.', ...
            accelPatternLabels{k}, size(feats.Location,1));
    end
    accelPts0 = [accelPts0; sel];
    rectangle('Position', roi, 'EdgeColor', clrAll(nLines+k,:), 'LineWidth', 1.5);
    plot(sel(:,1), sel(:,2), 's', 'Color', clrAll(nLines+k,:), ...
         'MarkerFaceColor', clrAll(nLines+k,:));
end
title('Tracking points placed'); drawnow;
exportgraphics(gcf, fullfile(outputDir,'01_tracked_points.png'), 'Resolution', 300);

%% ----- TRACK: NCC for line points, KLT for accelerometer points ---------
% NCC = normalized cross-correlation template matching. Good for points
%       placed at arbitrary locations along a line (no need for a corner).
% KLT = Kanade-Lucas-Tomasi. Good for corner-like features in a ROI.
tracker = vision.PointTracker('MaxBidirectionalError', 1, ...
    'NumPyramidLevels', 3, 'BlockSize', [31 31]);
initialize(tracker, accelPts0, firstFrame);

nFest    = floor(v.Duration * fs) + 5;
nTotal   = nStructPts + nPat * N;
trkY     = nan(nFest, nTotal);
trkY(1,:) = [pts0(:,2); accelPts0(:,2)].';
curPos    = pts0;
searchPad = 10;

i = 1; hWait = waitbar(0, 'Tracking video...');
while hasFrame(v)
    i = i + 1;
    fr     = readFrame(v);
    frGray = im2gray(fr);

    % --- NCC for structural line points ---
    for j = 1:nStructPts
        tmpl = templates{j};
        [tH, tW] = size(tmpl);
        cx = round(curPos(j,1));  cy = round(curPos(j,2));
        sH = halfPatch + searchPad;
        sr1 = max(1, cy-sH);  sr2 = min(size(frGray,1), cy+sH);
        sc1 = max(1, cx-sH);  sc2 = min(size(frGray,2), cx+sH);
        region = frGray(sr1:sr2, sc1:sc2);
        if size(region,1) < tH || size(region,2) < tW
            trkY(i,j) = NaN;  continue;
        end
        C = normxcorr2(tmpl, region);
        [~, mi] = max(C(:));
        [pr, pc] = ind2sub(size(C), mi);
        newY = sr1 + pr - tH + floor(tH/2);
        newX = sc1 + pc - tW + floor(tW/2);
        curPos(j,:) = [newX, newY];
        trkY(i, j)  = newY;
    end

    % --- KLT for accelerometer-ROI points ---
    [pos, valid] = tracker(fr);
    pos(~valid, :) = NaN;
    trkY(i, nStructPts+1:end) = pos(:,2).';

    if mod(i,30) == 0, waitbar(min(i/nFest,1), hWait); end
end
close(hWait);
nF   = i;
trkY = trkY(1:nF, :);
t    = (0:nF-1).' / fs;
fprintf('Tracked %d frames (%.2f s)\n', nF, t(end));

%% ----- TIME HISTORIES (Y displacement in px) ----------------------------
Y = trkY - trkY(1, :);
Y = fillmissing(Y, 'linear', 1, 'EndValues', 'nearest');
Y = detrend(Y, 1);

Y_struct = Y(:, 1:nStructPts);                % nF x (nLines*N)
Y_accAll = Y(:, nStructPts+1:end);            % nF x (nPat*N), per KLT point
Y_acc    = zeros(nF, nPat);                   % nF x nPat, one signal per ROI
for k = 1:nPat
    Y_acc(:, k) = mean(Y_accAll(:, (k-1)*N + (1:N)), 2, 'omitnan');
end

%% ----- PSD: periodogram on active window, no zero padding ---------------
tMask  = t >= analysisWindow(1) & t <= analysisWindow(2);
nWin   = sum(tMask);
win    = hann(nWin, 'periodic');
df     = fs / nWin;
fprintf('Analysis window: %.2f-%.2f s (%d samples, df = %.4f Hz)\n', ...
    analysisWindow, nWin, df);

[Pxx,    f] = periodogram(Y_struct(tMask, :), win, nWin, fs);
[Pacc,   ~] = periodogram(Y_acc(tMask, :),    win, nWin, fs);
[Pacc_pp,~] = periodogram(Y_accAll(tMask, :), win, nWin, fs);

% Per-line average PSD (structural lines: avg of N points; accel: 1 signal)
Pline = zeros(numel(f), nLines + nPat);
for k = 1:nLines
    Pline(:, k) = mean(Pxx(:, (k-1)*N + (1:N)), 2);
end
for k = 1:nPat
    Pline(:, nLines+k) = Pacc(:, k);
end
% Global average across structural lines only
Pglobal = mean(Pline(:, 1:nLines), 2);

%% ----- BIN-SNAP PEAK DETECTION inside magBand ---------------------------
bandIdx = f >= magBand(1) & f <= magBand(2);
fb      = f(bandIdx);
peakHz  = @(P) bandArgMax(P, bandIdx, fb);

% Per-point peaks: line points + per-KLT-point in each accel ROI
peakStructPP = arrayfun(@(j) peakHz(Pxx(:,j)),     1:nStructPts).';
peakAccPP    = arrayfun(@(j) peakHz(Pacc_pp(:,j)), 1:(nPat*N)).';
peakAllPts   = [peakStructPP; peakAccPP];

% Per-line peak of the averaged PSD
peakPerLine  = arrayfun(@(k) peakHz(Pline(:,k)), 1:(nLines+nPat)).';

% Global peak (peak of the cross-portion averaged PSD)
peakGlobal   = peakHz(Pglobal);

% Overall pooled statistics
muPeak = mean(peakAllPts);
sdPeak = std(peakAllPts);
errPct = 100 * abs(peakGlobal - accelTargetHz) / accelTargetHz;

fprintf('\n===== PEAK SUMMARY =====\n');
fprintf('  df (resolution)         : %.4f Hz  (+/- %.4f Hz uncertainty floor)\n', df, df/2);
fprintf('  Global peak (avg PSD)   : %.4f Hz  (vs accel %.2f Hz: %.2f%% error)\n', ...
    peakGlobal, accelTargetHz, errPct);
fprintf('  Pooled per-point peaks  : %.4f +/- %.4f Hz   (n=%d)\n', ...
    muPeak, sdPeak, numel(peakAllPts));

%% ----- CROSS-SPECTRUM (magnitude + phase) between accel ROIs ------------
patSig = detrend(Y_acc(tMask, :), 1);
Yfft   = fft(patSig .* win, nWin);
nFpos  = floor(nWin/2) + 1;
Yfft   = Yfft(1:nFpos, :);
f_cs   = (0:nFpos-1).' * fs / nWin;
scale  = 1 / (fs * sum(win.^2));

pairs     = nchoosek(1:nPat, 2);
nPairs    = size(pairs, 1);
CPSD      = zeros(nFpos, nPairs);
pairNames = strings(1, nPairs);
for k = 1:nPairs
    a = pairs(k,1);  b = pairs(k,2);
    Gxy = Yfft(:,a) .* conj(Yfft(:,b)) * scale;
    Gxy(2:end-1) = 2 * Gxy(2:end-1);          % single-sided
    CPSD(:, k)   = Gxy;
    pairNames(k) = sprintf('%s - %s', accelPatternLabels{a}, accelPatternLabels{b});
end
CPSDmag     = abs(CPSD);
CPSDphase   = rad2deg(angle(CPSD));
CPSDmag_avg = mean(CPSDmag, 2);

%% ----- TIME-DOMAIN CROSS-CORRELATION between accel ROIs -----------------
% No extra filtering: the magnification step already band-limits the signal
maxLag = round(min(1.5, 0.5*(analysisWindow(2)-analysisWindow(1))) * fs);
lagSec = (-maxLag:maxLag).' / fs;
xcMat     = zeros(2*maxLag+1, nPairs);
bestLag   = zeros(nPairs, 1);
bestPhase = zeros(nPairs, 1);
for k = 1:nPairs
    a = pairs(k,1);  b = pairs(k,2);
    s1 = patSig(:,a) - mean(patSig(:,a));
    s2 = patSig(:,b) - mean(patSig(:,b));
    xc = xcorr(s1, s2, maxLag, 'normalized');
    xcMat(:, k) = xc;
    [~, im] = max(abs(xc));
    bestLag(k)   = lagSec(im);
    bestPhase(k) = wrapTo180(360 * peakGlobal * bestLag(k));
end

fprintf('\nTime-domain cross-correlation (lag and phase at %.3f Hz):\n', peakGlobal);
for k = 1:nPairs
    fprintf('  %-20s  lag = %+0.4f s   phase = %+0.1f deg\n', ...
        pairNames(k), bestLag(k), bestPhase(k));
end

%% ----- SAVE TABLES (CSV) ------------------------------------------------
groupLabels = [string(lineLabels), string(accelPatternLabels)];

% Time histories (structural + per-ROI averaged accelerometer)
colNames = strings(1, nStructPts + nPat);
for k = 1:nLines, for j = 1:N
        colNames((k-1)*N + j) = sprintf('%s_P%d', matlab.lang.makeValidName(lineLabels{k}), j);
    end, end
for k = 1:nPat
    colNames(nStructPts + k) = string(matlab.lang.makeValidName(accelPatternLabels{k}));
end
THtbl = array2table([t, Y_struct, Y_acc], 'VariableNames', ['time_s' cellstr(colNames)]);
writetable(THtbl, fullfile(outputDir,'time_histories_Y_px.csv'));

% Per-point peaks (all 50/80 of them) with their group label
ppGroup = strings(numel(peakAllPts), 1);
for k = 1:nLines
    ppGroup((k-1)*N + (1:N)) = string(lineLabels{k});
end
for k = 1:nPat
    ppGroup(nStructPts + (k-1)*N + (1:N)) = string(accelPatternLabels{k});
end
writetable(table(ppGroup, peakAllPts, 'VariableNames', {'Group','PeakHz'}), ...
    fullfile(outputDir,'per_point_peaks.csv'));

% Summary: per-line peaks + overall pooled
peakSummary = table(groupLabels(:), peakPerLine, ...
    'VariableNames', {'Group','AvgPSDPeakHz'});
overallRow = table("OVERALL_POOLED", muPeak, 'VariableNames', {'Group','AvgPSDPeakHz'});
peakSummary = [peakSummary; overallRow];
peakSummary.Mean_pm_Std = strings(height(peakSummary), 1);
peakSummary.Mean_pm_Std(end) = sprintf('%.4f +/- %.4f Hz (n=%d)', muPeak, sdPeak, numel(peakAllPts));
writetable(peakSummary, fullfile(outputDir,'peak_summary.csv'));

% Cross-correlation results
writetable(table(pairNames(:), bestLag, bestPhase, ...
    'VariableNames', {'Pair','Lag_s','Phase_deg_at_peak'}), ...
    fullfile(outputDir,'xcorr_lag_phase.csv'));

%% ===== PLOTS =============================================================
saveHQ = @(fig,name) exportgraphics(fig, fullfile(outputDir,[name '.png']), 'Resolution', 300);

%% ----- Plot: time histories per group -----------------------------------
fig = figure('Name','Time histories','Color','w','Position',[80 80 1300 800]);
tl = tiledlayout(nLines+nPat, 1, 'TileSpacing','compact','Padding','compact');
title(tl, 'Y-displacement time histories', 'FontWeight','bold');
for k = 1:nLines
    nexttile; hold on; grid on; box on;
    sig = Y_struct(:, (k-1)*N + (1:N));
    plot(t, sig);
    xline(analysisWindow(1), 'k:', 'HandleVisibility','off');
    xline(analysisWindow(2), 'k:', 'HandleVisibility','off');
    ylabel('Y-disp [px]');
    title(lineLabels{k}, 'FontWeight','normal');
end
for k = 1:nPat
    nexttile; hold on; grid on; box on;
    plot(t, Y_acc(:,k));
    xline(analysisWindow(1), 'k:', 'HandleVisibility','off');
    xline(analysisWindow(2), 'k:', 'HandleVisibility','off');
    ylabel('Y-disp [px]');
    title(accelPatternLabels{k}, 'FontWeight','normal');
end
xlabel('Time [s]');
saveHQ(fig, '02_time_histories');

%% ----- Plot: per-line PSDs (individual dotted + bold average) ----------
for k = 1:(nLines + nPat)
    fig = figure('Name', sprintf('PSD %s', groupLabels{k}), 'Color', 'w', ...
        'Position', [80 80 1100 600]);
    ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    if k <= nLines
        Pcols = Pxx(:, (k-1)*N + (1:N));
    else
        Pcols = Pacc_pp(:, (k-nLines-1)*N + (1:N));
    end
    nCurves = size(Pcols, 2);

    % Individual point PSDs
    hInd = semilogy(ax, f, max(Pcols, eps), ':', 'LineWidth', 1.0, 'Color',[0.55 0.55 0.55]);

    % Bold black average
    Pavg = mean(Pcols, 2);
    hAvg = semilogy(ax, f, max(Pavg, eps), 'k-', 'LineWidth', 3.0);

    % Bin-snap peak from the average
    pkHz = peakHz(Pavg);
    hPk  = xline(ax, pkHz, 'r--', sprintf('video peak = %.3f Hz', pkHz), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal','Color',[0.85 0.1 0.1]);

    % Accelerometer reference
    hRef = xline(ax, accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
        'LineWidth', 2.0, 'Color',[0.05 0.3 0.85], 'LabelOrientation','horizontal', ...
        'LabelVerticalAlignment','bottom');

    set(ax,'YScale','log'); xlim(ax, spectraXLim);
    xlabel(ax,'Frequency [Hz]'); ylabel(ax,'PSD [px^{2}/Hz]');
    title(ax, sprintf('%s   |   peak of avg PSD = %.3f Hz   (df = %.3f Hz)', ...
        groupLabels{k}, pkHz, df), 'FontWeight','normal');
    legend(ax, [hInd(1), hAvg, hPk, hRef], ...
        {sprintf('Individual PSDs (n=%d)', nCurves), 'Average PSD (bold)', ...
         'Detected peak (video)','Accelerometer reference'}, 'Location','northeast');
    saveHQ(fig, sprintf('03_PSD_%s', matlab.lang.makeValidName(groupLabels{k})));
end

%% ----- Plot: overall pooled peak (mean +/- std of all per-point peaks) ---
fig = figure('Name','Overall pooled peak','Color','w','Position',[80 80 1100 600]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% Dots aligned vertically per group (no jitter)
xc = 1:(nLines+nPat);
hSc = [];
for k = 1:nLines
    fp = peakStructPP((k-1)*N + (1:N));
    hh = scatter(ax, k*ones(N,1), fp, 50, [0.30 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor','k', 'LineWidth', 0.3, ...
        'HandleVisibility','off');
    if isempty(hSc), hSc = hh; end
end
for k = 1:nPat
    fp = peakAccPP((k-1)*N + (1:N));
    scatter(ax, (nLines+k)*ones(N,1), fp, 50, [0.30 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor','k', 'LineWidth', 0.3, ...
        'HandleVisibility','off');
end
hSc.HandleVisibility = 'on';
hSc.DisplayName = sprintf('Individual point peaks (n = %d)', numel(peakAllPts));

% Pooled mean +/- std as a horizontal band
xR = [0.5, nLines+nPat+0.5];
fill(ax, [xR fliplr(xR)], [muPeak-sdPeak muPeak-sdPeak muPeak+sdPeak muPeak+sdPeak], ...
    [0.10 0.30 0.75], 'FaceAlpha', 0.18, 'EdgeColor','none', ...
    'DisplayName', sprintf('Overall mean \\pm 1\\sigma'));
yline(ax, muPeak, '-', 'LineWidth', 2.5, 'Color',[0.10 0.30 0.75], ...
    'DisplayName', sprintf('Overall mean = %.4f Hz', muPeak));

% Accelerometer reference
yline(ax, accelTargetHz, '--', 'LineWidth', 2.0, 'Color',[0.75 0.10 0.10], ...
    'DisplayName', sprintf('Accel = %.2f Hz', accelTargetHz));

ylim(ax, [min([peakAllPts; accelTargetHz])-0.1*df-0.05, ...
          max([peakAllPts; accelTargetHz])+0.1*df+0.05]);
xlim(ax, xR);
set(ax, 'XTick', xc, 'XTickLabel', groupLabels);
xlabel(ax,'Tracking group');  ylabel(ax,'Peak frequency [Hz]');
title(ax, sprintf('OVERALL: %.4f \\pm %.4f Hz   (n=%d)   error vs accel: %.2f%%', ...
    muPeak, sdPeak, numel(peakAllPts), 100*abs(muPeak-accelTargetHz)/accelTargetHz), ...
    'FontWeight','bold');
legend(ax,'Location','southoutside','Orientation','horizontal');
saveHQ(fig, '04_overall_pooled_peak');

%% ----- Plot: cross-spectrum magnitude + phase ---------------------------
fig = figure('Name','Cross spectrum','Color','w','Position',[60 60 1300 800]);
tl  = tiledlayout(2,1,'TileSpacing','compact','Padding','compact');
title(tl, sprintf('Cross-spectrum between %d accelerometer ROIs', nPat), 'FontWeight','bold');

nexttile; hold on; grid on; box on;
for k = 1:nPairs
    plot(f_cs, 10*log10(max(CPSDmag(:,k), eps)), ':', 'LineWidth', 1.0, ...
        'DisplayName', pairNames(k));
end
plot(f_cs, 10*log10(max(CPSDmag_avg, eps)), 'k-', 'LineWidth', 3.0, ...
    'DisplayName','Average |CPSD|');
xline(peakGlobal, 'r--', sprintf('peak = %.3f Hz', peakGlobal), 'HandleVisibility','off');
xline(accelTargetHz, '-', sprintf('accel = %.2f Hz', accelTargetHz), ...
    'Color',[0.05 0.3 0.85], 'LineWidth', 2, 'HandleVisibility','off');
ylabel('|CPSD| [dB]'); xlim(spectraXLim);
title('Cross-spectral magnitude','FontWeight','normal');
legend('Location','northeastoutside');

nexttile; hold on; grid on; box on;
for k = 1:nPairs
    plot(f_cs, CPSDphase(:,k), '-', 'LineWidth', 1.0, 'DisplayName', pairNames(k));
end
xline(peakGlobal, 'r--', 'HandleVisibility','off');
yline(  0, 'k--','in-phase','HandleVisibility','off');
yline(180, 'k:', 'out-of-phase','HandleVisibility','off');
yline(-180,'k:', 'HandleVisibility','off');
xlabel('Frequency [Hz]'); ylabel('Phase [\circ]');
ylim([-180 180]); xlim(spectraXLim);
title('Cross-spectral phase','FontWeight','normal');
legend('Location','northeastoutside');
saveHQ(fig, '05_cross_spectrum');

%% ----- Plot: time-domain xcorr ------------------------------------------
fig = figure('Name','Time-domain xcorr','Color','w','Position',[80 80 1200 600]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
cmap = lines(nPairs);
for k = 1:nPairs
    plot(ax, lagSec, xcMat(:,k), '-', 'LineWidth', 1.3, 'Color', cmap(k,:), ...
        'DisplayName', sprintf('%s (lag=%+0.3fs, \\phi=%+0.1f\\circ)', ...
        pairNames(k), bestLag(k), bestPhase(k)));
end
xline(ax, 0, 'k--', 'HandleVisibility','off');
xlabel(ax,'Lag [s]');  ylabel(ax,'Normalized cross-correlation');
title(ax, sprintf('Time-domain xcorr (no additional filtering, window %.1f-%.1f s)', ...
    analysisWindow), 'FontWeight','normal');
legend(ax,'Location','northeastoutside');
saveHQ(fig, '06_time_xcorr');

%% ----- Final summary printout -------------------------------------------
fprintf('\n===== SAVED TO: %s =====\n', outputDir);
fprintf('Headline result for the paper:\n');
fprintf('  Video natural frequency  : %.4f Hz  (peak of cross-portion avg PSD)\n', peakGlobal);
fprintf('  Pooled per-point average : %.4f +/- %.4f Hz   (n=%d, df=%.3f Hz)\n', ...
    muPeak, sdPeak, numel(peakAllPts), df);
fprintf('  Accelerometer reference  : %.2f Hz  (error %.2f%%)\n', accelTargetHz, errPct);

%% ===== LOCAL FUNCTION ===================================================
function fpk = bandArgMax(P, bandIdx, fb)
    % Bin-snap peak: frequency of the bin with maximum dB-PSD in the band.
    % No interpolation; uncertainty floor is df/2.
    Pb = P(bandIdx);
    if isempty(Pb), fpk = NaN; return; end
    [~, im] = max(10*log10(max(Pb, eps)));
    fpk = fb(im);
end

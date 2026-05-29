clear; clc; close all;

%% ----- USER CONFIG -------------------------------------------------------
videoFile = "D:\dronebasedVMM\Deep-Motion-Mag-Pytorch-main\Rightview-lab-final\9.5-10.5Hz\output_frames_P02m_30_fl9.5_fh10.5_fs60.0_n4_butter.mp4";

lineLabels         = {'1st landing','2nd landing'};
accelPatternLabels = {'acc0','acc1','acc2','acc3'};
nPointsPerLine     = 5;

% Modal analysis settings
analysisWindow = [0 5];      % seconds, the segment where the structure rings
magBand        = [9 11];     % Hz, peak search band (matches the magnification)
accelTargetHz  = 10.2;       % Hz, accelerometer ground-truth frequency
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
nLines = numel(lineLabels);
nPat   = numel(accelPatternLabels);
N      = nPointsPerLine;

figure('Name','Setup', 'Color', 'w','NumberTitle','off');
imshow(firstFrame); hold on;

lineCoords = zeros(nLines, 4);   % [x1 y1 x2 y2]
for k = 1:nLines
    title(sprintf('Draw line %d/%d: %s', k, nLines, lineLabels{k}));
    h = drawline('Color', 'w', 'LineWidth', 2);
    wait(h);
    lineCoords(k,:) = [h.Position(1,:) h.Position(2,:)];
end

% Distribute N points along each line, grab a small NCC template per point
nStructPts = nLines * N;
pts0       = zeros(nStructPts, 2);
lineTmpl   = cell(nStructPts, 1);
halfPatch  = 20;     % half-size of the line-point template (px) -> 41x41
for k = 1:nLines
    s  = linspace(0, 1, N).';
    xs = lineCoords(k,1) + s * (lineCoords(k,3) - lineCoords(k,1));
    ys = lineCoords(k,2) + s * (lineCoords(k,4) - lineCoords(k,2));
    idx = (k-1)*N + (1:N);
    pts0(idx,:) = [xs ys];
    plot(xs, ys, 'o', 'Color', 'w', 'MarkerFaceColor', 'w');
    for j = idx
        cx = round(pts0(j,1));  cy = round(pts0(j,2));
        r1 = max(1, cy-halfPatch);  r2 = min(size(firstGray,1), cy+halfPatch);
        c1 = max(1, cx-halfPatch);  c2 = min(size(firstGray,2), cx+halfPatch);
        lineTmpl{j} = firstGray(r1:r2, c1:c2);
    end
end

%% ----- DRAW ACCELEROMETER RECTANGLES (one NCC template per ROI) ---------
% You draw a rectangle directly on each accelerometer / its housing.
% The rectangle's pixels ARE the template; it is tracked frame-to-frame
% with normalized cross-correlation. One signal per accelerometer.
accRect   = zeros(nPat, 4);   % [x y w h]
accTmpl   = cell(nPat, 1);
accPos0   = zeros(nPat, 2);   % template center at frame 1
accTH     = zeros(nPat, 1);   % template height
accTW     = zeros(nPat, 1);   % template width
for k = 1:nPat
    title(sprintf('Draw rectangle %d/%d on %s', k, nPat, accelPatternLabels{k}));
    hR = drawrectangle('Color', 'r', 'LineWidth', 1.8);
    wait(hR);
    accRect(k,:) = hR.Position;
    roi = round(accRect(k,:));
    % clamp to image bounds
    x1 = max(1, roi(1));               y1 = max(1, roi(2));
    x2 = min(size(firstGray,2), x1 + max(roi(3)-1, 0));
    y2 = min(size(firstGray,1), y1 + max(roi(4)-1, 0));
    accTmpl{k} = firstGray(y1:y2, x1:x2);
    [accTH(k), accTW(k)] = size(accTmpl{k});
    accPos0(k,:) = [(x1+x2)/2, (y1+y2)/2];
    rectangle('Position', [x1 y1 x2-x1+1 y2-y1+1], 'EdgeColor', 'r', 'LineWidth', 1.5);
    plot(accPos0(k,1), accPos0(k,2), 's', 'Color', 'r', ...
         'MarkerFaceColor', 'r', 'MarkerSize', 10);
end
title('Tracking templates placed'); drawnow;
exportgraphics(gcf, fullfile(outputDir,'01_tracked_points.png'), 'Resolution', 300);

%% ----- TRACK (NCC for every point and every accel rectangle) -----------
nFest    = floor(v.Duration * fs) + 5;
nTotal   = nStructPts + nPat;
trkY     = nan(nFest, nTotal);
trkY(1, 1:nStructPts) = pts0(:,2).';
trkY(1, nStructPts+1:end) = accPos0(:,2).';

curLinePos = pts0;
curAccPos  = accPos0;
searchPad     = 10;     % extra pixels around the line-point template
accSearchPad  = 30;     % extra pixels around the accelerometer template

i = 1; hWait = waitbar(0, 'Tracking video...');
while hasFrame(v)
    i = i + 1;
    fr     = readFrame(v);
    frGray = im2gray(fr);

    % --- line points (small templates) ---
    for j = 1:nStructPts
        tmpl = lineTmpl{j};
        [tH, tW] = size(tmpl);
        cx = round(curLinePos(j,1));  cy = round(curLinePos(j,2));
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
        curLinePos(j,:) = [newX, newY];
        trkY(i, j) = newY;
    end

    % --- accelerometer rectangles (one large template each) ---
    for k = 1:nPat
        tmpl = accTmpl{k};
        tH = accTH(k);  tW = accTW(k);
        cx = round(curAccPos(k,1));  cy = round(curAccPos(k,2));
        sr1 = max(1, round(cy - tH/2 - accSearchPad));
        sr2 = min(size(frGray,1), round(cy + tH/2 + accSearchPad));
        sc1 = max(1, round(cx - tW/2 - accSearchPad));
        sc2 = min(size(frGray,2), round(cx + tW/2 + accSearchPad));
        region = frGray(sr1:sr2, sc1:sc2);
        if size(region,1) < tH || size(region,2) < tW
            trkY(i, nStructPts+k) = NaN;  continue;
        end
        C = normxcorr2(tmpl, region);
        [~, mi] = max(C(:));
        [pr, pc] = ind2sub(size(C), mi);
        newY = sr1 + pr - tH + floor(tH/2);
        newX = sc1 + pc - tW + floor(tW/2);
        curAccPos(k,:) = [newX, newY];
        trkY(i, nStructPts+k) = newY;
    end

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

Y_struct = Y(:, 1:nStructPts);            % line points
Y_acc    = Y(:, nStructPts+1:end);        % one signal per accelerometer ROI

%% ----- PSD: periodogram on active window, no zero padding ---------------
tMask = t >= analysisWindow(1) & t <= analysisWindow(2);
nWin  = sum(tMask);
win   = hann(nWin, 'periodic');
df    = fs / nWin;
fprintf('Analysis window: %.2f-%.2f s (%d samples, df = %.4f Hz)\n', ...
    analysisWindow, nWin, df);

[Pxx,  f] = periodogram(Y_struct(tMask, :), win, nWin, fs);
[Pacc, ~] = periodogram(Y_acc(tMask, :),    win, nWin, fs);

% Per-line average PSD (lines = mean of N points; accel = the single signal)
Pline = zeros(numel(f), nLines + nPat);
for k = 1:nLines
    Pline(:, k) = mean(Pxx(:, (k-1)*N + (1:N)), 2);
end
for k = 1:nPat
    Pline(:, nLines+k) = Pacc(:, k);
end
% Cross-portion average over structural lines
Pglobal = mean(Pline(:, 1:nLines), 2);

%% ----- BIN-SNAP PEAK DETECTION inside magBand ---------------------------
bandIdx = f >= magBand(1) & f <= magBand(2);
fb      = f(bandIdx);
peakHz  = @(P) bandArgMax(P, bandIdx, fb);

% Per-point peaks: line points + one peak per accelerometer ROI
peakStructPP = arrayfun(@(j) peakHz(Pxx(:,j)),  1:nStructPts).';
peakAccPP    = arrayfun(@(j) peakHz(Pacc(:,j)), 1:nPat).';
peakAllPts   = [peakStructPP; peakAccPP];

% Per-line averaged-PSD peak (line-by-line OR one per accel)
peakPerLine  = arrayfun(@(k) peakHz(Pline(:,k)), 1:(nLines+nPat)).';

% Global peak (peak of the cross-portion average over structural lines)
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

%% ----- ROI SIGNAL QUALITY (out-of-band SNR) -----------------------------
% A good modal response sticks well above the OUT-OF-BAND noise floor.
% In-band median is not a useful baseline here because the magnification
% flattens the response across magBand, so peak/in-band-median is ~5 even
% for clean ROIs. Compare instead to median of bins OUTSIDE magBand
% but inside the plot range.
outBandIdx = ((f >= spectraXLim(1)) & (f <  magBand(1))) | ...
             ((f >  magBand(2))     & (f <= spectraXLim(2)));
fprintf('\n===== ROI SIGNAL QUALITY (peak vs out-of-band noise floor) =====\n');
snrFloorDb = 20;     % below this we flag the ROI as untrustworthy (~100x linear)
for k = 1:nPat
    Pk     = Pacc(:, k);
    pkVal  = max(Pk(bandIdx));
    nzVal  = median(Pk(outBandIdx));
    snrLin = pkVal / max(nzVal, eps);
    snrDb  = 10 * log10(snrLin);
    flag   = '';
    if snrDb < snrFloorDb
        flag = '  <-- LOW SNR, peak likely an artefact';
    end
    fprintf('  %-6s  peak/noise = %8.1f (%5.1f dB)   peak = %.3f Hz%s\n', ...
        accelPatternLabels{k}, snrLin, snrDb, peakPerLine(nLines+k), flag);
end

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

% Time histories (structural + per-ROI accelerometer)
colNames = strings(1, nStructPts + nPat);
for k = 1:nLines
    for j = 1:N
        colNames((k-1)*N + j) = sprintf('%s_P%d', matlab.lang.makeValidName(lineLabels{k}), j);
    end
end
for k = 1:nPat
    colNames(nStructPts + k) = string(matlab.lang.makeValidName(accelPatternLabels{k}));
end
THtbl = array2table([t, Y_struct, Y_acc], 'VariableNames', ['time_s' cellstr(colNames)]);
writetable(THtbl, fullfile(outputDir,'time_histories_Y_px.csv'));

% Per-point peaks
ppGroup = strings(numel(peakAllPts), 1);
for k = 1:nLines
    ppGroup((k-1)*N + (1:N)) = string(lineLabels{k});
end
for k = 1:nPat
    ppGroup(nStructPts + k) = string(accelPatternLabels{k});
end
writetable(table(ppGroup, peakAllPts, 'VariableNames', {'Group','PeakHz'}), ...
    fullfile(outputDir,'per_point_peaks.csv'));

% Summary: per-line + overall pooled
peakSummary = table(groupLabels(:), peakPerLine, 'VariableNames', {'Group','AvgPSDPeakHz'});
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

%% ----- Plot: PSDs per group --------------------------------------------
% Lines: N individual point PSDs (dotted) + bold black average
% Accelerometers: one PSD (bold black) - single template per ROI
for k = 1:(nLines + nPat)
    fig = figure('Name', sprintf('PSD %s', groupLabels{k}), 'Color', 'w', ...
        'Position', [80 80 1100 600]);
    ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    if k <= nLines
        Pcols = Pxx(:, (k-1)*N + (1:N));
        hInd  = semilogy(ax, f, max(Pcols, eps), ':', 'LineWidth', 1.0, 'Color',[0.55 0.55 0.55]);
        Pavg  = mean(Pcols, 2);
        nCurves = size(Pcols, 2);
        indLabel = sprintf('Individual PSDs (n=%d)', nCurves);
    else
        Pavg = Pacc(:, k-nLines);
        hInd = [];                     % no individual curves for accel
        indLabel = '';
    end

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
    if isempty(hInd)
        legend(ax, [hAvg, hPk, hRef], ...
            {'PSD (bold)', 'Detected peak (video)','Accelerometer reference'}, ...
            'Location','northeast');
    else
        legend(ax, [hInd(1), hAvg, hPk, hRef], ...
            {indLabel, 'Average PSD (bold)', 'Detected peak (video)','Accelerometer reference'}, ...
            'Location','northeast');
    end
    saveHQ(fig, sprintf('03_PSD_%s', matlab.lang.makeValidName(groupLabels{k})));
end

%% ----- Plot: overall pooled peak (mean +/- std of all per-point peaks) ---
fig = figure('Name','Overall pooled peak','Color','w','Position',[80 80 1100 600]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

% Dots aligned vertically per group (no jitter)
xc  = 1:(nLines+nPat);
hSc = [];
for k = 1:nLines
    fp = peakStructPP((k-1)*N + (1:N));
    hh = scatter(ax, k*ones(N,1), fp, 50, [0.30 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor','k', 'LineWidth', 0.3, ...
        'HandleVisibility','off');
    if isempty(hSc), hSc = hh; end
end
for k = 1:nPat
    scatter(ax, (nLines+k), peakAccPP(k), 50, [0.30 0.45 0.75], 'filled', ...
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

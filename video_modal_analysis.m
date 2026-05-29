clear; clc; close all;

%% ----- USER CONFIG -------------------------------------------------------
videoFile = "D:\dronebasedVMM\Deep-Motion-Mag-Pytorch-main\Rightview-lab-final\9.5-10.5Hz\output_frames_P02m_30_fl9.5_fh10.5_fs60.0_n4_butter.mp4";

lineLabels         = {'1st landing','2nd landing'};
accelPatternLabels = {'acc0','acc1','acc2','acc3'};
nPointsPerLine     = 5;

% Modal analysis settings
analysisWindow = [0 5];       % seconds, the segment where the structure rings
magBand        = [9 11];      % Hz, peak search band (matches the magnification)
accelTargetHz  = 10.2;        % Hz, accelerometer ground-truth frequency
spectraXLim    = [8 19];      % Hz, x-axis for spectra plots

% Mode-shape overlay visual magnification.
%   'auto'  -> scaled so the maximum deflection is ~5% of the image height
%   number  -> e.g. 400 (matches the example you sent)
modeShapeVisualMag = 'auto';

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

lineVerts = cell(nLines, 1);   % each cell: Mk x 2 array of polyline vertices
snapTolPx = 25;                % snap a new vertex onto an existing one within this radius
for k = 1:nLines
    title(sprintf('Draw polyline %d/%d (%s): click vertices, double-click to finish', ...
        k, nLines, lineLabels{k}));
    h = drawpolyline('Color', 'w', 'LineWidth', 2);
    wait(h);
    verts = h.Position;             % M x 2 array [x y]
    % Snap each endpoint of this polyline to any previously-drawn vertex
    % within snapTolPx pixels, so adjacent portions can share corners.
    for prev = 1:k-1
        for ii = [1, size(verts,1)]
            d = hypot(lineVerts{prev}(:,1) - verts(ii,1), ...
                      lineVerts{prev}(:,2) - verts(ii,2));
            [dmin, jj] = min(d);
            if dmin <= snapTolPx
                verts(ii,:) = lineVerts{prev}(jj,:);
            end
        end
    end
    lineVerts{k} = verts;
end

% Distribute N points along each polyline's ARC LENGTH and grab an NCC
% template per point. A multi-segment polyline lets one portion follow a
% staircase corner; adjacent portions share endpoints when the user
% (or the snap above) places them on the same pixel.
nStructPts = nLines * N;
pts0       = zeros(nStructPts, 2);
lineTmpl   = cell(nStructPts, 1);
halfPatch  = 20;     % half-size of the line-point template (px) -> 41x41
for k = 1:nLines
    verts = lineVerts{k};
    segLen = hypot(diff(verts(:,1)), diff(verts(:,2)));
    cumL   = [0; cumsum(segLen)];
    totalL = cumL(end);
    sTarget = linspace(0, totalL, N).';
    xs = interp1(cumL, verts(:,1), sTarget);
    ys = interp1(cumL, verts(:,2), sTarget);
    idx = (k-1)*N + (1:N);
    pts0(idx,:) = [xs ys];
    plot(verts(:,1), verts(:,2), '-', 'Color', 'w', 'LineWidth', 1.5);
    plot(xs, ys, 'o', 'Color', 'w', 'MarkerFaceColor', 'w');
    for j = idx
        cx = round(pts0(j,1));  cy = round(pts0(j,2));
        r1 = max(1, cy-halfPatch);  r2 = min(size(firstGray,1), cy+halfPatch);
        c1 = max(1, cx-halfPatch);  c2 = min(size(firstGray,2), cx+halfPatch);
        lineTmpl{j} = firstGray(r1:r2, c1:c2);
    end
end

%% ----- DRAW ACCELEROMETER RECTANGLES (one NCC template per ROI) ---------
accRect = zeros(nPat, 4);
accTmpl = cell(nPat, 1);
accPos0 = zeros(nPat, 2);
accTH   = zeros(nPat, 1);
accTW   = zeros(nPat, 1);
for k = 1:nPat
    title(sprintf('Draw rectangle %d/%d on %s', k, nPat, accelPatternLabels{k}));
    hR = drawrectangle('Color', 'r', 'LineWidth', 1.8);
    wait(hR);
    accRect(k,:) = hR.Position;
    roi = round(accRect(k,:));
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
% Both X and Y are stored so a 2-D operational mode shape can be drawn.
nFest  = floor(v.Duration * fs) + 5;
nTotal = nStructPts + nPat;
trkX   = nan(nFest, nTotal);
trkY   = nan(nFest, nTotal);
trkX(1, 1:nStructPts)         = pts0(:,1).';
trkY(1, 1:nStructPts)         = pts0(:,2).';
trkX(1, nStructPts+1:end)     = accPos0(:,1).';
trkY(1, nStructPts+1:end)     = accPos0(:,2).';

curLinePos   = pts0;
curAccPos    = accPos0;
searchPad    = 10;     % extra pixels around the line-point template
accSearchPad = 30;     % extra pixels around the accelerometer template

i = 1; hWait = waitbar(0, 'Tracking video...');
while hasFrame(v)
    i = i + 1;
    fr     = readFrame(v);
    frGray = im2gray(fr);

    % --- line points ---
    for j = 1:nStructPts
        tmpl = lineTmpl{j};
        [tH, tW] = size(tmpl);
        cx = round(curLinePos(j,1));  cy = round(curLinePos(j,2));
        sH = halfPatch + searchPad;
        sr1 = max(1, cy-sH);  sr2 = min(size(frGray,1), cy+sH);
        sc1 = max(1, cx-sH);  sc2 = min(size(frGray,2), cx+sH);
        region = frGray(sr1:sr2, sc1:sc2);
        if size(region,1) < tH || size(region,2) < tW
            continue;
        end
        C = normxcorr2(tmpl, region);
        [~, mi] = max(C(:));
        [pr, pc] = ind2sub(size(C), mi);
        newY = sr1 + pr - tH + floor(tH/2);
        newX = sc1 + pc - tW + floor(tW/2);
        curLinePos(j,:) = [newX, newY];
        trkX(i, j) = newX;
        trkY(i, j) = newY;
    end

    % --- accelerometer rectangles ---
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
            continue;
        end
        C = normxcorr2(tmpl, region);
        [~, mi] = max(C(:));
        [pr, pc] = ind2sub(size(C), mi);
        newY = sr1 + pr - tH + floor(tH/2);
        newX = sc1 + pc - tW + floor(tW/2);
        curAccPos(k,:) = [newX, newY];
        trkX(i, nStructPts+k) = newX;
        trkY(i, nStructPts+k) = newY;
    end

    if mod(i,30) == 0, waitbar(min(i/nFest,1), hWait); end
end
close(hWait);
nF   = i;
trkX = trkX(1:nF, :);
trkY = trkY(1:nF, :);
t    = (0:nF-1).' / fs;
fprintf('Tracked %d frames (%.2f s)\n', nF, t(end));

%% ----- TIME HISTORIES (X and Y displacement in px) ---------------------
X = trkX - trkX(1, :);
Y = trkY - trkY(1, :);
X = fillmissing(X, 'linear', 1, 'EndValues', 'nearest');
Y = fillmissing(Y, 'linear', 1, 'EndValues', 'nearest');
X = detrend(X, 1);
Y = detrend(Y, 1);

X_struct = X(:, 1:nStructPts);
Y_struct = Y(:, 1:nStructPts);
Y_acc    = Y(:, nStructPts+1:end);

%% ----- PSD: periodogram on active window, no zero padding ---------------
% Modal frequency identification uses Y (the dominant direction of the
% bandpass-magnified mode). The mode shape itself is computed in 2-D.
tMask = t >= analysisWindow(1) & t <= analysisWindow(2);
nWin  = sum(tMask);
win   = hann(nWin, 'periodic');
df    = fs / nWin;
fprintf('Analysis window: %.2f-%.2f s (%d samples, df = %.4f Hz)\n', ...
    analysisWindow, nWin, df);

[Pxx,  f] = periodogram(Y_struct(tMask, :), win, nWin, fs);
[Pacc, ~] = periodogram(Y_acc(tMask, :),    win, nWin, fs);

Pline = zeros(numel(f), nLines + nPat);
for k = 1:nLines
    Pline(:, k) = mean(Pxx(:, (k-1)*N + (1:N)), 2);
end
for k = 1:nPat
    Pline(:, nLines+k) = Pacc(:, k);
end
Pglobal = mean(Pline(:, 1:nLines), 2);

%% ----- BIN-SNAP PEAK DETECTION inside magBand ---------------------------
bandIdx = f >= magBand(1) & f <= magBand(2);
fb      = f(bandIdx);
peakHz  = @(P) bandArgMax(P, bandIdx, fb);

peakStructPP = arrayfun(@(j) peakHz(Pxx(:,j)),  1:nStructPts).';
peakAccPP    = arrayfun(@(j) peakHz(Pacc(:,j)), 1:nPat).';
peakAllPts   = [peakStructPP; peakAccPP];
peakPerLine  = arrayfun(@(k) peakHz(Pline(:,k)), 1:(nLines+nPat)).';
peakGlobal   = peakHz(Pglobal);

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
outBandIdx = ((f >= spectraXLim(1)) & (f <  magBand(1))) | ...
             ((f >  magBand(2))     & (f <= spectraXLim(2)));
fprintf('\n===== ROI SIGNAL QUALITY (peak vs out-of-band noise floor) =====\n');
snrFloorDb = 20;
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
    Gxy(2:end-1) = 2 * Gxy(2:end-1);
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

%% ----- OPERATIONAL MODE SHAPE at peakGlobal (2-D) ----------------------
% At a single dominant frequency, the displacement of each point is
%     d_x(t) = Re( Ux * exp(-i*2*pi*f*t) )
%     d_y(t) = Re( Uy * exp(-i*2*pi*f*t) )
% where Ux, Uy are the complex FFT coefficients at the peak bin.
% We render the snapshot at the phase phi* that maximises the L2 norm
% of the deflection over all line points:
%     phi* = 0.5 * angle( sum_j (Ux_j^2 + Uy_j^2) )
% i.e. the instant of maximum collective displacement.
Xwin = X_struct(tMask, :) .* win;
Ywin = Y_struct(tMask, :) .* win;
Xspec = fft(Xwin, nWin);
Yspec = fft(Ywin, nWin);
fMode = (0:floor(nWin/2)).' * fs / nWin;
[~, ipk] = min(abs(fMode - peakGlobal));

% Normalize the FFT coefficient to a physical amplitude in pixels.
% For a sinusoid of amplitude A and a window w, the bin magnitude is
%     |X[k]| = A * sum(w) / 2
% so multiplying by 2/sum(w) recovers the amplitude. After this scaling,
% |U_j| is the per-cycle displacement of point j in pixels, and the
% snapshot dx, dy below is also in pixels -> visMag = 1 means real motion.
windowGain = sum(win) / 2;
Ux = Xspec(ipk, :).' / windowGain;
Uy = Yspec(ipk, :).' / windowGain;

phiStar = 0.5 * angle( sum(Ux.^2 + Uy.^2) );
dx = real(Ux * exp(-1i * phiStar));
dy = real(Uy * exp(-1i * phiStar));

% Visual magnification: 'auto' targets ~5% of image height for the
% largest deflection, otherwise use the user-supplied number.
maxDef = max(hypot(dx, dy));
if isnumeric(modeShapeVisualMag)
    visMag = modeShapeVisualMag;
else
    targetDef = 0.05 * size(firstFrame, 1);
    visMag    = max(1, round(targetDef / max(maxDef, eps)));
end
xRef = pts0(:,1);  yRef = pts0(:,2);
xDef = xRef + visMag * dx;
yDef = yRef + visMag * dy;
fprintf('\nMode shape at %.3f Hz: max real deflection = %.3f px, visual mag = x%d\n', ...
    peakGlobal, maxDef, visMag);

%% ----- SAVE TABLES (CSV) ------------------------------------------------
groupLabels = [string(lineLabels), string(accelPatternLabels)];

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

ppGroup = strings(numel(peakAllPts), 1);
for k = 1:nLines
    ppGroup((k-1)*N + (1:N)) = string(lineLabels{k});
end
for k = 1:nPat
    ppGroup(nStructPts + k) = string(accelPatternLabels{k});
end
writetable(table(ppGroup, peakAllPts, 'VariableNames', {'Group','PeakHz'}), ...
    fullfile(outputDir,'per_point_peaks.csv'));

peakSummary = table(groupLabels(:), peakPerLine, 'VariableNames', {'Group','AvgPSDPeakHz'});
peakSummary = [peakSummary; table("OVERALL_POOLED", muPeak, ...
    'VariableNames', {'Group','AvgPSDPeakHz'})];
peakSummary.Mean_pm_Std = strings(height(peakSummary), 1);
peakSummary.Mean_pm_Std(end) = sprintf('%.4f +/- %.4f Hz (n=%d)', muPeak, sdPeak, numel(peakAllPts));
writetable(peakSummary, fullfile(outputDir,'peak_summary.csv'));

writetable(table(pairNames(:), bestLag, bestPhase, ...
    'VariableNames', {'Pair','Lag_s','Phase_deg_at_peak'}), ...
    fullfile(outputDir,'xcorr_lag_phase.csv'));

% Mode shape (reference + deflected pixel coordinates per point)
msGroup = strings(nStructPts, 1);
for k = 1:nLines
    msGroup((k-1)*N + (1:N)) = string(lineLabels{k});
end
writetable(table(msGroup, xRef, yRef, dx, dy, xDef, yDef, ...
    'VariableNames', {'Group','xRef_px','yRef_px','dx_px','dy_px','xDef_px','yDef_px'}), ...
    fullfile(outputDir,'mode_shape_at_peak.csv'));

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
for k = 1:(nLines + nPat)
    fig = figure('Name', sprintf('PSD %s', groupLabels{k}), 'Color', 'w', ...
        'Position', [80 80 1100 600]);
    ax = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    if k <= nLines
        Pcols   = Pxx(:, (k-1)*N + (1:N));
        hInd    = semilogy(ax, f, max(Pcols, eps), ':', 'LineWidth', 1.0, 'Color',[0.55 0.55 0.55]);
        Pavg    = mean(Pcols, 2);
        nCurves = size(Pcols, 2);
        indLabel = sprintf('Individual PSDs (n=%d)', nCurves);
    else
        Pavg = Pacc(:, k-nLines);
        hInd = [];
        indLabel = '';
    end

    hAvg = semilogy(ax, f, max(Pavg, eps), 'k-', 'LineWidth', 3.0);

    pkHz = peakHz(Pavg);
    hPk  = xline(ax, pkHz, 'r--', sprintf('video peak = %.3f Hz', pkHz), ...
        'LineWidth', 1.6, 'LabelOrientation','horizontal','Color',[0.85 0.1 0.1]);
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

%% ----- Plot: overall pooled peak ----------------------------------------
fig = figure('Name','Overall pooled peak','Color','w','Position',[80 80 1100 600]);
ax  = axes(fig); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
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

xR = [0.5, nLines+nPat+0.5];
fill(ax, [xR fliplr(xR)], [muPeak-sdPeak muPeak-sdPeak muPeak+sdPeak muPeak+sdPeak], ...
    [0.10 0.30 0.75], 'FaceAlpha', 0.18, 'EdgeColor','none', ...
    'DisplayName', sprintf('Overall mean \\pm 1\\sigma'));
yline(ax, muPeak, '-', 'LineWidth', 2.5, 'Color',[0.10 0.30 0.75], ...
    'DisplayName', sprintf('Overall mean = %.4f Hz', muPeak));
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

%% ----- Plot: 2-D MODE SHAPE OVERLAY on the first frame -----------------
% Three things per portion are drawn on the first frame:
%   1. drawn line   - yellow dashed - endpoints of the user-drawn line
%   2. ref          - solid colored, thin - through tracked initial points
%   3. def (xVis)   - solid colored, thick + square markers - through the
%                     deflected positions at the moment of maximum collective
%                     displacement (visually magnified)
% Thin colored "drift" segments connect each ref point to its def position
% so the displacement vector of every point is visible.
fig = figure('Name','Mode shape overlay','Color','k','Position',[60 60 1500 900]);
ax  = axes(fig);
imshow(firstFrame, 'Parent', ax); hold(ax, 'on');

% High-contrast saturated colors per portion. Each element of the plot
% is drawn with a black halo first so it stays readable on light AND
% dark backgrounds (steel / glass / shadow all show up in your clips).
portionColors = {[1.00 0.20 0.85], ... % magenta
                 [0.10 0.85 1.00], ... % cyan
                 [0.30 1.00 0.15], ... % lime
                 [1.00 0.55 0.10]};    % orange
hLegEntries = gobjects(0);
legNames    = strings(0);

for k = 1:nLines
    idx = (k-1)*N + (1:N);
    cl  = portionColors{mod(k-1, numel(portionColors)) + 1};

    % 1. Drift segments (bottom layer) - per-point displacement vectors
    for j = idx
        plot(ax, [xRef(j) xDef(j)], [yRef(j) yDef(j)], '-', ...
            'Color', [cl 0.55], 'LineWidth', 1.0, 'HandleVisibility','off');
    end

    % 2. Drawn polyline = reference shape (black halo + yellow dashed)
    plot(ax, lineVerts{k}(:,1), lineVerts{k}(:,2), '-', ...
        'Color', 'k', 'LineWidth', 4.5, 'HandleVisibility','off');
    hDr = plot(ax, lineVerts{k}(:,1), lineVerts{k}(:,2), '--', ...
        'Color', [1 1 0], 'LineWidth', 2.5);

    % 3. Reference sample-point markers (white circles outlined in black)
    plot(ax, xRef(idx), yRef(idx), 'o', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'w', 'MarkerSize', 7, ...
        'LineWidth', 0.8, 'HandleVisibility','off');

    % 4. Deflected polyline (black halo + saturated color on top)
    plot(ax, xDef(idx), yDef(idx), '-', ...
        'Color', 'k', 'LineWidth', 5.5, 'HandleVisibility','off');
    hDe = plot(ax, xDef(idx), yDef(idx), '-', 'Color', cl, 'LineWidth', 3.2);
    plot(ax, xDef(idx), yDef(idx), 's', ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', cl, 'MarkerSize', 9, ...
        'LineWidth', 0.8, 'HandleVisibility','off');

    hLegEntries = [hLegEntries, hDr, hDe];
    legNames    = [legNames, ...
        string(lineLabels{k}) + " --- reference (drawn)", ...
        string(lineLabels{k}) + sprintf(" --- deflected (x%d)", visMag)];
end

ttl = title(ax, sprintf('Mode shape overlay  -  %.3f Hz  (x%d visual)', peakGlobal, visMag), ...
    'Color', 'w', 'FontSize', 13, 'FontWeight','bold');
lg = legend(hLegEntries, legNames, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns', nLines, ...
    'TextColor','w', 'Color','k', 'EdgeColor',[0.6 0.6 0.6], 'FontSize', 10);
saveHQ(fig, '07_mode_shape_overlay');

%% ----- Final summary printout -------------------------------------------
fprintf('\n===== SAVED TO: %s =====\n', outputDir);
fprintf('Headline result for the paper:\n');
fprintf('  Video natural frequency  : %.4f Hz  (peak of cross-portion avg PSD)\n', peakGlobal);
fprintf('  Pooled per-point average : %.4f +/- %.4f Hz   (n=%d, df=%.3f Hz)\n', ...
    muPeak, sdPeak, numel(peakAllPts), df);
fprintf('  Accelerometer reference  : %.2f Hz  (error %.2f%%)\n', accelTargetHz, errPct);
fprintf('  Mode shape rendered at   : %.3f Hz   (visual magnification x%d)\n', peakGlobal, visMag);

%% ===== LOCAL FUNCTION ===================================================
function fpk = bandArgMax(P, bandIdx, fb)
    Pb = P(bandIdx);
    if isempty(Pb), fpk = NaN; return; end
    [~, im] = max(10*log10(max(Pb, eps)));
    fpk = fb(im);
end

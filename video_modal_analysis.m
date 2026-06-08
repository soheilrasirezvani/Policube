clear; clc; close all;

%% ----- USER CONFIG -------------------------------------------------------
videoFile = "D:\dronebasedVMM\Deep-Motion-Mag-Pytorch-main\Rightview-lab-final\9.5-10.5Hz\output_frames_P02m_30_fl9.5_fh10.5_fs60.0_n4_butter.mp4";

lineLabels         = {'1st landing'};
accelPatternLabels = {'acc0','acc1','acc2','acc3'};
nPointsPerLine     = 20;

% Modal analysis settings
analysisWindow = [0 5];       % seconds, the segment where the structure rings
magBand        = [9 11];      % Hz, peak search band (matches the magnification)
accelTargetHz  = 10.2;        % Hz, accelerometer ground-truth frequency
spectraXLim    = [8 19];      % Hz, x-axis for spectra plots

% Mode-shape overlay visual magnification.
%   'auto'  -> scaled so the maximum deflection is ~5% of the image height
%   number  -> e.g. 400 (matches the example you sent)
modeShapeVisualMag = 'auto';

% Tracking robustness
useSubpixelNCC     = true;     % parabolic refinement of the NCC peak (sub-pixel resolution)
applyContrastBoost = true;     % CLAHE on the frame before NCC (helps low-contrast ROIs)
nccConfidenceMin   = 0.30;     % NCC peak below this -> hold previous position
halfPatch          = 25;       % line-point template half-size (px) -> (2*halfPatch+1)^2
searchPad          = 12;       % search padding around line-point template (px)

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
if applyContrastBoost
    firstGray = adapthisteq(firstGray, 'ClipLimit', 0.02);
end
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
% Every polyline VERTEX is guaranteed to be a tracked point (the joints
% between sections are structurally meaningful for the mode shape).
% Each portion then gets max(N, M_k) total points: the M_k vertices plus
% extra points distributed along the segments proportionally to length.
linePtCount = zeros(nLines, 1);
lineIdx     = cell(nLines, 1);
ptr = 0;
for k = 1:nLines
    M = size(lineVerts{k}, 1);
    Nk = max(N, M);
    linePtCount(k) = Nk;
    lineIdx{k}     = ptr + (1:Nk);
    ptr = ptr + Nk;
end
nStructPts = ptr;
pts0       = zeros(nStructPts, 2);
lineTmpl   = cell(nStructPts, 1);
isJoint    = false(nStructPts, 1);   % true where pts0(j) is a polyline vertex
for k = 1:nLines
    verts  = lineVerts{k};
    M      = size(verts, 1);
    Nk     = linePtCount(k);
    segLen = hypot(diff(verts(:,1)), diff(verts(:,2)));
    cumL   = [0; cumsum(segLen)];
    totalL = cumL(end);

    if Nk == M
        sTarget = cumL;             % only vertices fit
    else
        % Distribute (Nk - M) extra points to segments in proportion to
        % their length, then place them evenly inside each segment.
        nExtra  = Nk - M;
        ideal   = nExtra * segLen / totalL;
        nPerSeg = floor(ideal);
        leftover = nExtra - sum(nPerSeg);
        [~, ord] = sort(ideal - nPerSeg, 'descend');
        nPerSeg(ord(1:leftover)) = nPerSeg(ord(1:leftover)) + 1;

        sTarget = cumL;             % anchor at every vertex
        for s = 1:numel(nPerSeg)
            if nPerSeg(s) > 0
                xs = linspace(cumL(s), cumL(s+1), nPerSeg(s)+2);
                sTarget = [sTarget; xs(2:end-1).'];   % interior of segment
            end
        end
        sTarget = sort(sTarget);
    end

    xs = interp1(cumL, verts(:,1), sTarget);
    ys = interp1(cumL, verts(:,2), sTarget);
    idx = lineIdx{k};
    pts0(idx,:) = [xs ys];

    % Mark which of these points coincide with a polyline vertex (joint).
    isJoint(idx) = ismembertol(sTarget, cumL, 1e-9);

    plot(verts(:,1), verts(:,2), '-', 'Color', 'w', 'LineWidth', 1.5);
    plot(xs, ys, 'o', 'Color', 'w', 'MarkerFaceColor', 'w');
    % Make joints visually distinct on the setup figure (larger diamond)
    plot(xs(isJoint(idx)), ys(isJoint(idx)), 'd', ...
        'Color', 'y', 'MarkerFaceColor', 'y', 'MarkerSize', 10);

    for j = idx
        cx = round(pts0(j,1));  cy = round(pts0(j,2));
        r1 = max(1, cy-halfPatch);  r2 = min(size(firstGray,1), cy+halfPatch);
        c1 = max(1, cx-halfPatch);  c2 = min(size(firstGray,2), cx+halfPatch);
        lineTmpl{j} = firstGray(r1:r2, c1:c2);
    end
end
fprintf('Structural points: %d total (%s) - joints anchored at every polyline vertex\n', ...
    nStructPts, strjoin(arrayfun(@(x) sprintf('%d',x), linePtCount(:).', 'UniformOutput', false), '+'));

%% ----- DRAW ACCELEROMETER RECTANGLES (KLT corner tracking) -------------
% Draw the FIRST rectangle freely. Its size [width x height] is then
% locked in for all subsequent ROIs - you only click the center of
% each remaining ROI and a same-size rectangle is placed there. This
% keeps every ROI dimensionally identical so per-ROI feature counts
% and noise statistics are comparable.
%
% KLT-trackable corner features are detected inside each ROI; the
% per-ROI averaged signal is the MEAN of the feature positions.
% vision.PointTracker is inherently sub-pixel (gradient-based optical
% flow) so no further refinement is needed.
nFeatsAccROI = 10;          % max KLT corners per ROI
accRect      = zeros(nPat, 4);
accPos0      = zeros(nPat, 2);     % mean of detected features (one per ROI)
accFeats0    = cell(nPat, 1);      % features inside each ROI, M_k x 2
for k = 1:nPat
    if k == 1
        title(sprintf(['Draw rectangle 1/%d on %s   ' ...
              '(its size will be reused for the remaining ROIs)'], ...
              nPat, accelPatternLabels{k}));
        hR = drawrectangle('Color', 'r', 'LineWidth', 1.8);
        wait(hR);
        accRect(k,:) = hR.Position;
        refW = accRect(k,3);  refH = accRect(k,4);
    else
        title(sprintf('Click CENTRE of ROI %d/%d on %s   (size locked to %d x %d px)', ...
              k, nPat, accelPatternLabels{k}, round(refW), round(refH)));
        [cx, cy] = ginput(1);
        accRect(k,:) = [cx - refW/2, cy - refH/2, refW, refH];
    end
    roi = round(accRect(k,:));
    roi(3:4) = max(roi(3:4), 5);
    roi(1) = max(1, roi(1));
    roi(2) = max(1, roi(2));
    roi(3) = min(roi(3), size(firstGray,2) - roi(1));
    roi(4) = min(roi(4), size(firstGray,1) - roi(2));

    feats = detectMinEigenFeatures(firstGray, 'ROI', roi, 'MinQuality', 0.005);
    nFound = size(feats.Location, 1);
    if nFound >= nFeatsAccROI
        [~, ord]      = sort(feats.Metric, 'descend');
        accFeats0{k}  = feats.Location(ord(1:nFeatsAccROI), :);
    elseif nFound > 0
        accFeats0{k}  = feats.Location;
        warning('ROI %s: only %d corners found (asked for %d).', ...
            accelPatternLabels{k}, nFound, nFeatsAccROI);
    else
        cx = roi(1) + roi(3)/2;  cy = roi(2) + roi(4)/2;
        accFeats0{k}  = [cx, cy];
        warning('ROI %s: no corners found - falling back to ROI center.', ...
            accelPatternLabels{k});
    end
    accPos0(k,:) = mean(accFeats0{k}, 1);
    rectangle('Position', roi, 'EdgeColor', 'r', 'LineWidth', 1.5);
    plot(accFeats0{k}(:,1), accFeats0{k}(:,2), '+', 'Color', 'r', ...
         'MarkerSize', 8, 'LineWidth', 1.5);
end

% Concatenate all KLT features and remember which rows belong to which ROI
accFeatIdx   = cell(nPat, 1);
ptr = 0;
for k = 1:nPat
    M = size(accFeats0{k}, 1);
    accFeatIdx{k} = ptr + (1:M);
    ptr = ptr + M;
end
allAccFeats0 = vertcat(accFeats0{:});

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

% Per-KLT-feature position history (one column per individual feature)
nAccFeats = size(allAccFeats0, 1);
trkXFeat  = nan(nFest, nAccFeats);
trkYFeat  = nan(nFest, nAccFeats);
trkXFeat(1,:) = allAccFeats0(:,1).';
trkYFeat(1,:) = allAccFeats0(:,2).';

curLinePos   = pts0;
curAccPos    = accPos0;
nccFailCount = zeros(nStructPts + nPat, 1);   % per-point low-confidence frame counter

% KLT tracker for accelerometer features (initialized on firstGray so
% it sees the same CLAHE-processed pixels as the per-frame inputs).
accTracker = vision.PointTracker('MaxBidirectionalError', 1, ...
    'NumPyramidLevels', 3, 'BlockSize', [31 31]);
initialize(accTracker, allAccFeats0, firstGray);

i = 1; hWait = waitbar(0, 'Tracking video...');
while hasFrame(v)
    i = i + 1;
    fr     = readFrame(v);
    frGray = im2gray(fr);
    if applyContrastBoost
        frGray = adapthisteq(frGray, 'ClipLimit', 0.02);
    end

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
        [pr_sub, pc_sub, maxC] = nccPeakSubpixel(C, useSubpixelNCC);
        if maxC < nccConfidenceMin
            % Low-confidence frame: hold previous position
            trkX(i, j) = trkX(i-1, j);
            trkY(i, j) = trkY(i-1, j);
            nccFailCount(j) = nccFailCount(j) + 1;
            continue;
        end
        newY = sr1 + pr_sub - tH + floor(tH/2);
        newX = sc1 + pc_sub - tW + floor(tW/2);
        curLinePos(j,:) = [newX, newY];
        trkX(i, j) = newX;
        trkY(i, j) = newY;
    end

    % --- accelerometer rectangles (KLT, per-feature + averaged) ---
    [posKLT, validKLT] = accTracker(frGray);
    % Per-feature: invalid -> NaN (will be filled by fillmissing later)
    xF = posKLT(:,1).';  yF = posKLT(:,2).';
    xF(~validKLT.') = NaN;  yF(~validKLT.') = NaN;
    trkXFeat(i,:) = xF;
    trkYFeat(i,:) = yF;
    for k = 1:nPat
        rows  = accFeatIdx{k};
        vMask = validKLT(rows);
        if any(vMask)
            curAccPos(k,:) = mean(posKLT(rows(vMask), :), 1);
            trkX(i, nStructPts+k) = curAccPos(k,1);
            trkY(i, nStructPts+k) = curAccPos(k,2);
        else
            trkX(i, nStructPts+k) = trkX(i-1, nStructPts+k);
            trkY(i, nStructPts+k) = trkY(i-1, nStructPts+k);
            nccFailCount(nStructPts+k) = nccFailCount(nStructPts+k) + 1;
        end
    end

    if mod(i,30) == 0, waitbar(min(i/nFest,1), hWait); end
end
close(hWait);
nF   = i;
trkX     = trkX(1:nF, :);
trkY     = trkY(1:nF, :);
trkXFeat = trkXFeat(1:nF, :);
trkYFeat = trkYFeat(1:nF, :);
t        = (0:nF-1).' / fs;
fprintf('Tracked %d frames (%.2f s)\n', nF, t(end));

% Report low-confidence frame counts per point/ROI (>5% triggers a flag)
flagThresh = 0.05 * (nF-1);
nFails     = nccFailCount;
if any(nFails > 0)
    fprintf('\n===== NCC LOW-CONFIDENCE FRAME COUNT (peak < %.2f) =====\n', nccConfidenceMin);
    for k = 1:nLines
        for j = 1:numel(lineIdx{k})
            idxJ = lineIdx{k}(j);
            if nFails(idxJ) > 0
                flag = '';
                if nFails(idxJ) > flagThresh
                    flag = sprintf('  <-- %.0f%% of frames held', 100*nFails(idxJ)/(nF-1));
                end
                fprintf('  %s pt %d  : %d frames%s\n', lineLabels{k}, j, nFails(idxJ), flag);
            end
        end
    end
    for k = 1:nPat
        n = nFails(nStructPts + k);
        if n > 0
            flag = '';
            if n > flagThresh
                flag = sprintf('  <-- %.0f%% of frames held', 100*n/(nF-1));
            end
            fprintf('  %s          : %d frames%s\n', accelPatternLabels{k}, n, flag);
        end
    end
end

%% ----- TIME HISTORIES (X and Y displacement in px) ---------------------
X = trkX - trkX(1, :);
Y = trkY - trkY(1, :);
X = fillmissing(X, 'linear', 1, 'EndValues', 'nearest');
Y = fillmissing(Y, 'linear', 1, 'EndValues', 'nearest');
X = detrend(X, 1);
Y = detrend(Y, 1);

% Displacements stay in pixels. The camera viewing angle makes
% mm-per-pixel vary across the frame (perspective), so a single
% scalar calibration would be wrong for every point except the one
% used to compute it. Report in pixels and let the reader interpret
% relative amplitudes.
X_struct = X(:, 1:nStructPts);
Y_struct = Y(:, 1:nStructPts);
Y_acc    = Y(:, nStructPts+1:end);

% Per-KLT-feature Y displacement (one column per individual feature).
% Used for the per-ROI mean +/- std statistic on the modal peak: every
% feature gives one peak-frequency estimate.
Y_accFeat = trkYFeat - trkYFeat(1, :);
Y_accFeat = fillmissing(Y_accFeat, 'linear', 1, 'EndValues', 'nearest');
Y_accFeat = detrend(Y_accFeat, 1);

%% ----- PSD: periodogram on active window, no zero padding ---------------
% Modal frequency identification uses Y (the dominant direction of the
% bandpass-magnified mode). The mode shape itself is computed in 2-D.
tMask = t >= analysisWindow(1) & t <= analysisWindow(2);
nWin  = sum(tMask);
win   = hann(nWin, 'periodic');
df    = fs / nWin;
fprintf('Analysis window: %.2f-%.2f s (%d samples, df = %.4f Hz)\n', ...
    analysisWindow, nWin, df);

[Pxx,      f] = periodogram(Y_struct(tMask, :),  win, nWin, fs);
[Pacc,     ~] = periodogram(Y_acc(tMask, :),     win, nWin, fs);
[Pacc_pp,  ~] = periodogram(Y_accFeat(tMask, :), win, nWin, fs);   % per KLT feature

Pline = zeros(numel(f), nLines + nPat);
for k = 1:nLines
    Pline(:, k) = mean(Pxx(:, lineIdx{k}), 2);
end
for k = 1:nPat
    Pline(:, nLines+k) = Pacc(:, k);
end
Pglobal = mean(Pline(:, 1:nLines), 2);

%% ----- BIN-SNAP PEAK DETECTION inside magBand ---------------------------
bandIdx = f >= magBand(1) & f <= magBand(2);
fb      = f(bandIdx);
peakHz  = @(P) bandArgMax(P, bandIdx, fb);

% One peak per individual measurement: every line point gets one peak,
% every KLT feature inside an accelerometer ROI gets one peak.
peakStructPP = arrayfun(@(j) peakHz(Pxx(:,j)),     1:nStructPts).';
peakFeatPP   = arrayfun(@(j) peakHz(Pacc_pp(:,j)), 1:nAccFeats).';
peakAllPts   = [peakStructPP; peakFeatPP];

% Peak of the averaged PSD per group (for the AvgPSDPeakHz column in the
% summary table). This is the "peak of the mean", separate from the
% mean of per-point peaks.
peakPerLine  = arrayfun(@(k) peakHz(Pline(:,k)), 1:(nLines+nPat)).';
peakGlobal   = peakHz(Pglobal);

% Per-group mean / std of the INDIVIDUAL measurements that belong to
% that group (line points for portions, KLT features for accel ROIs).
peakGroupMean = nan(nLines+nPat, 1);
peakGroupStd  = nan(nLines+nPat, 1);
peakGroupN    = zeros(nLines+nPat, 1);
for k = 1:nLines
    fp = peakStructPP(lineIdx{k});
    peakGroupMean(k) = mean(fp, 'omitnan');
    peakGroupStd(k)  = std(fp,  0, 'omitnan');
    peakGroupN(k)    = numel(fp);
end
for k = 1:nPat
    fp = peakFeatPP(accFeatIdx{k});
    peakGroupMean(nLines+k) = mean(fp, 'omitnan');
    peakGroupStd(nLines+k)  = std(fp,  0, 'omitnan');
    peakGroupN(nLines+k)    = numel(fp);
end

% Backwards-compat alias used by Plot 7 (one number per accelerometer ROI
% from the averaged-signal PSD - still useful as a reference dot).
peakAccPP = arrayfun(@(j) peakHz(Pacc(:,j)), 1:nPat).';

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

%% ----- ROI TRACKER NOISE FLOOR (time-domain) ----------------------------
% Take the QUIESCENT portions of the signal (before the impulse and after
% the ring-down) as a direct measurement of the tracker noise floor in
% pixels. Then compare to the peak-to-peak amplitude during the active
% period. If the noise floor is comparable to the peak amplitude, you
% ARE seeing the structural decay - it is just buried in tracker noise.
quietBefore = t < 1.0;
quietAfter  = t > 3.5 & t <= 5.0;
quietMask   = quietBefore | quietAfter;
activeMask  = t >= 1.0 & t <= 3.0;
fprintf('\n===== TRACKER NOISE FLOOR vs MODAL SIGNAL (time domain) =====\n');
fprintf('  quiet windows: t<1.0 s and 3.5<t<=5.0 s  |  active: 1.0<=t<=3.0 s\n');
for k = 1:nPat
    s   = Y_acc(:, k);
    nFlo = std(s(quietMask), 0);
    pk   = max(abs(s(activeMask)));
    snrA = pk / max(nFlo, eps);
    flag = '';
    if snrA < 5
        flag = '  <-- tracker noise dominates';
    end
    fprintf('  %-6s  noise floor = %.3f px   active peak = %.3f px   ratio = %5.1fx%s\n', ...
        accelPatternLabels{k}, nFlo, pk, snrA, flag);
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
dx = real(Ux * exp(-1i * phiStar));        % per-point deflection in pixels
dy = real(Uy * exp(-1i * phiStar));

% Visual magnification: 'auto' targets ~5% of image height for the
% largest pixel deflection, otherwise use the user-supplied number.
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
fprintf('\nMode shape at %.3f Hz: max deflection = %.3f px, visual mag = x%d\n', ...
    peakGlobal, maxDef, visMag);

%% ----- SAVE TABLES (CSV) ------------------------------------------------
groupLabels = [string(lineLabels), string(accelPatternLabels)];

colNames = strings(1, nStructPts + nPat);
for k = 1:nLines
    Nk = linePtCount(k);
    for j = 1:Nk
        suffix = "_P" + j;
        if isJoint(lineIdx{k}(j)), suffix = "_J" + j; end   % J flags a joint
        colNames(lineIdx{k}(j)) = string(matlab.lang.makeValidName(lineLabels{k})) + suffix;
    end
end
for k = 1:nPat
    colNames(nStructPts + k) = string(matlab.lang.makeValidName(accelPatternLabels{k}));
end
THtbl = array2table([t, Y_struct, Y_acc], 'VariableNames', ['time_s' cellstr(colNames)]);
writetable(THtbl, fullfile(outputDir,'time_histories_Y_px.csv'));

% Per-measurement peak: one row per line point and one row per KLT
% feature (NOT one per accelerometer ROI). Numel matches peakAllPts.
ppGroup = strings(numel(peakAllPts), 1);
for k = 1:nLines
    ppGroup(lineIdx{k}) = string(lineLabels{k});
end
for k = 1:nPat
    ppGroup(nStructPts + accFeatIdx{k}) = string(accelPatternLabels{k});
end
writetable(table(ppGroup, peakAllPts, 'VariableNames', {'Group','PeakHz'}), ...
    fullfile(outputDir,'per_point_peaks.csv'));

% Summary: per-group Mean_pm_Std comes from the INDIVIDUAL measurements
% in that group (line points for portions, KLT features for accel ROIs).
% AvgPSDPeakHz is the peak of the per-group averaged PSD ("peak of mean")
% which is a different estimator and listed alongside for cross-check.
nGroupRows = nLines + nPat;
ms = strings(nGroupRows + 1, 1);
nv = zeros(nGroupRows + 1, 1);
for kk = 1:nGroupRows
    ms(kk) = sprintf('%.4f +/- %.4f Hz', peakGroupMean(kk), peakGroupStd(kk));
    nv(kk) = peakGroupN(kk);
end
ms(end) = sprintf('%.4f +/- %.4f Hz', muPeak, sdPeak);
nv(end) = numel(peakAllPts);

peakSummary = table([groupLabels(:); "OVERALL_POOLED"], ...
                    [peakPerLine;       muPeak], ...
                    nv, ms, ...
                    'VariableNames', {'Group','AvgPSDPeakHz','nMeasurements','Mean_pm_Std'});
writetable(peakSummary, fullfile(outputDir,'peak_summary.csv'));

writetable(table(pairNames(:), bestLag, bestPhase, ...
    'VariableNames', {'Pair','Lag_s','Phase_deg_at_peak'}), ...
    fullfile(outputDir,'xcorr_lag_phase.csv'));

% Mode shape: physical deflection per point in mm, plus the pixel
% coordinates used to render the (magnified) deflected polyline.
msGroup = strings(nStructPts, 1);
isJointCol = isJoint(:);
for k = 1:nLines
    msGroup(lineIdx{k}) = string(lineLabels{k});
end
writetable(table(msGroup, isJointCol, xRef, yRef, dx, dy, xDef, yDef, ...
    'VariableNames', {'Group','IsJoint','xRef_px','yRef_px','dx_px','dy_px','xDef_px_visAmp','yDef_px_visAmp'}), ...
    fullfile(outputDir,'mode_shape_at_peak.csv'));

%% ===== PLOTS =============================================================
saveHQ = @(fig,name) exportgraphics(fig, fullfile(outputDir,[name '.png']), 'Resolution', 300);

%% ----- Plot: time histories per group -----------------------------------
fig = figure('Name','Time histories','Color','w','Position',[80 80 1300 800]);
tl = tiledlayout(nLines+nPat, 1, 'TileSpacing','compact','Padding','compact');
title(tl, 'Y-displacement time histories', 'FontWeight','bold');
for k = 1:nLines
    nexttile; hold on; grid on; box on;
    sig = Y_struct(:, lineIdx{k});
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
        Pcols   = Pxx(:, lineIdx{k});
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
    fp = peakStructPP(lineIdx{k});
    hh = scatter(ax, k*ones(numel(fp),1), fp, 50, [0.30 0.45 0.75], 'filled', ...
        'MarkerFaceAlpha', 0.55, 'MarkerEdgeColor','k', 'LineWidth', 0.3, ...
        'HandleVisibility','off');
    if isempty(hSc), hSc = hh; end
end
for k = 1:nPat
    fp = peakFeatPP(accFeatIdx{k});       % one peak per KLT feature
    scatter(ax, (nLines+k)*ones(numel(fp),1), fp, 50, [0.30 0.45 0.75], 'filled', ...
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
ylabel('|CPSD| [dB ref px^{2}/Hz]'); xlim(spectraXLim);
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
% Envelope style: for each portion both extremes of the modal cycle are
% drawn (positive snapshot at phi* and negative snapshot at phi*+pi),
% joined at the polyline endpoints to close the envelope, with a light
% portion-colour fill between them.
%   Filled markers  = positive half-cycle (Re(U exp(-i phi*)))
%   Unfilled markers= negative half-cycle (-1 * above)
% The thin white dashed centerline is the user-drawn reference polyline
% so the envelope can be read as "structure swings between these two
% colored lines around the dashed centerline".
fig = figure('Name','Mode shape overlay','Color','w','Position',[60 60 1500 900]);
ax  = axes(fig);
imshow(firstFrame, 'Parent', ax); hold(ax, 'on');

% Per-portion colors. Matches the user's example envelope figure.
portionColors = {[0.20 0.50 1.00], ... % blue
                 [0.20 0.80 0.20], ... % green
                 [1.00 0.55 0.10], ... % orange
                 [0.85 0.10 0.10]};    % red
hLegEntries = gobjects(0);
legNames    = strings(0);

for k = 1:nLines
    idx = lineIdx{k};
    cl  = portionColors{mod(k-1, numel(portionColors)) + 1};
    jointMask = isJoint(idx);

    % Pixel deflections at this portion
    xRefK = xRef(idx);  yRefK = yRef(idx);
    dxK   = dx(idx);    dyK   = dy(idx);

    % Positive (+phi*) and negative (-phi*) envelope positions
    xPos = xRefK + visMag * dxK;
    yPos = yRefK + visMag * dyK;
    xNeg = xRefK - visMag * dxK;
    yNeg = yRefK - visMag * dyK;

    % 1. Filled envelope polygon (positive forward, negative reversed)
    xPoly = [xPos; flipud(xNeg)];
    yPoly = [yPos; flipud(yNeg)];
    patch(ax, xPoly, yPoly, cl, 'FaceAlpha', 0.18, 'EdgeColor', 'none', ...
        'HandleVisibility', 'off');

    % 2. Reference (drawn) centerline - thin white dashed with black halo
    plot(ax, lineVerts{k}(:,1), lineVerts{k}(:,2), '-', ...
        'Color', 'k', 'LineWidth', 2.8, 'HandleVisibility', 'off');
    hDr = plot(ax, lineVerts{k}(:,1), lineVerts{k}(:,2), '--', ...
        'Color', 'w', 'LineWidth', 1.6);

    % 3. Envelope polylines (black halo + portion color) - same color as
    %    the requested style and connected at the endpoints.
    plot(ax, xPos, yPos, '-', 'Color', 'k', 'LineWidth', 4.2, 'HandleVisibility', 'off');
    hPos = plot(ax, xPos, yPos, '-', 'Color', cl, 'LineWidth', 2.6);
    plot(ax, xNeg, yNeg, '-', 'Color', 'k', 'LineWidth', 4.2, 'HandleVisibility', 'off');
    plot(ax, xNeg, yNeg, '-', 'Color', cl, 'LineWidth', 2.6, 'HandleVisibility', 'off');

    % 4. Closing segments at start / end (link positive and negative)
    plot(ax, [xPos(1) xNeg(1)],   [yPos(1) yNeg(1)],   '-', ...
        'Color', cl, 'LineWidth', 2.6, 'HandleVisibility', 'off');
    plot(ax, [xPos(end) xNeg(end)], [yPos(end) yNeg(end)], '-', ...
        'Color', cl, 'LineWidth', 2.6, 'HandleVisibility', 'off');

    % 5. Markers per node
    %    Positive half cycle: filled circles, portion color, black edge
    %    Negative half cycle: open circles, portion color edge, white fill
    plot(ax, xPos(~jointMask), yPos(~jointMask), 'o', ...
        'MarkerFaceColor', cl, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 6, 'LineWidth', 0.7, 'HandleVisibility','off');
    plot(ax, xNeg(~jointMask), yNeg(~jointMask), 'o', ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', cl, ...
        'MarkerSize', 6, 'LineWidth', 1.4, 'HandleVisibility','off');

    %    Joints rendered slightly larger so polyline vertices stand out
    plot(ax, xPos(jointMask), yPos(jointMask), 'o', ...
        'MarkerFaceColor', cl, 'MarkerEdgeColor', 'k', ...
        'MarkerSize', 10, 'LineWidth', 0.9, 'HandleVisibility','off');
    plot(ax, xNeg(jointMask), yNeg(jointMask), 'o', ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', cl, ...
        'MarkerSize', 10, 'LineWidth', 1.6, 'HandleVisibility','off');

    hLegEntries = [hLegEntries, hPos];
    legNames    = [legNames, ...
        string(lineLabels{k}) + sprintf(" envelope (x%d)", visMag)];
end

% Global legend entries for the positive / negative half-cycle markers
hPosMk = plot(ax, NaN, NaN, 'o', 'MarkerFaceColor', [0.4 0.4 0.4], ...
    'MarkerEdgeColor', 'k', 'MarkerSize', 7, 'LineWidth', 0.8);
hNegMk = plot(ax, NaN, NaN, 'o', 'MarkerFaceColor', 'w', ...
    'MarkerEdgeColor', [0.4 0.4 0.4], 'MarkerSize', 7, 'LineWidth', 1.4);
hLegEntries = [hLegEntries, hDr, hPosMk, hNegMk];
legNames    = [legNames, "reference (drawn)", ...
    "filled marker = +half cycle", "open marker = -half cycle"];

ttl = title(ax, sprintf('Mode shape overlay  -  %.3f Hz  (envelope x%d)', peakGlobal, visMag), ...
    'Color', 'k', 'FontSize', 13, 'FontWeight','bold');
lg = legend(hLegEntries, legNames, 'Location','northoutside', ...
    'Orientation','horizontal', 'NumColumns', max(nLines, 3), ...
    'TextColor','k', 'Color','w', 'EdgeColor',[0.3 0.3 0.3], 'FontSize', 10);
saveHQ(fig, '07_mode_shape_overlay');

%% ----- Final summary printout -------------------------------------------
fprintf('\n===== SAVED TO: %s =====\n', outputDir);
fprintf('Headline result for the paper:\n');
fprintf('  Video natural frequency  : %.4f Hz  (peak of cross-portion avg PSD)\n', peakGlobal);
fprintf('  Pooled per-point average : %.4f +/- %.4f Hz   (n=%d, df=%.3f Hz)\n', ...
    muPeak, sdPeak, numel(peakAllPts), df);
fprintf('  Accelerometer reference  : %.2f Hz  (error %.2f%%)\n', accelTargetHz, errPct);
fprintf('  Mode shape rendered at   : %.3f Hz   (max deflection %.3f px, visual x%d)\n', ...
    peakGlobal, maxDef, visMag);

%% ===== LOCAL FUNCTIONS ==================================================
function fpk = bandArgMax(P, bandIdx, fb)
    Pb = P(bandIdx);
    if isempty(Pb), fpk = NaN; return; end
    [~, im] = max(10*log10(max(Pb, eps)));
    fpk = fb(im);
end

function [pr, pc, maxC] = nccPeakSubpixel(C, useSubpixel)
    % Locate the maximum of an NCC surface and (optionally) refine it to
    % sub-pixel accuracy with a 3-point parabolic fit along each axis.
    [maxC, mi] = max(C(:));
    [pr0, pc0] = ind2sub(size(C), mi);
    pr = pr0;  pc = pc0;
    if ~useSubpixel
        return;
    end
    if pr0 > 1 && pr0 < size(C, 1)
        a = C(pr0-1, pc0);  b = C(pr0, pc0);  c = C(pr0+1, pc0);
        denom = a - 2*b + c;
        if abs(denom) > eps
            shift = 0.5 * (a - c) / denom;
            pr = pr0 + max(min(shift, 0.5), -0.5);
        end
    end
    if pc0 > 1 && pc0 < size(C, 2)
        a = C(pr0, pc0-1);  b = C(pr0, pc0);  c = C(pr0, pc0+1);
        denom = a - 2*b + c;
        if abs(denom) > eps
            shift = 0.5 * (a - c) / denom;
            pc = pc0 + max(min(shift, 0.5), -0.5);
        end
    end
end

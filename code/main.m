%% Channel Coding Project – incremental redundancy
% COMM B504 – Spring 2025
% -------------------------------------------------------------------------
%  Authors: Mahmoud Elfil, Omar Emad, Youssef Sabry
%  Date   : 15-May-2025
% -------------------------------------------------------------------------

run(fullfile('..','startup.m'));       % add paths + cd code
clear; clc; close all;

%% 0) USER SETTINGS --------------------------------------------------------
aviFile        = fullfile('data','highway.avi');  % original video
p_vals         = logspace(log10(1e-4), log10(2e-1), 15); % FULL sweep
traceback      = 35;                               % Viterbi traceback
infoLen        = 1024;                             % bits / packet
motherRate     = [171 133];                        % generators (octal)

% puncturing patterns – X followed by Y (per brief)
punct8_9 = [1 1 1 1  0 1 1 1  1 0 0 0  1 0 0 0];
punct4_5 = [1 1 1 1  1 1 1 1  1 0 0 0  1 0 0 0];
punct2_3 = [1 1 1 1  1 1 1 1  1 0 1 0  1 0 1 0];
punct1_2 = ones(1,2);

rateTable = {punct8_9, punct4_5, punct2_3, punct1_2};
rateName  = {'8_9','4_5','2_3','1_2'};            % for look-ups

% showcase list exactly as required in PDF
demo_p      = [1e-3 1e-3 1e-3  1e-1 1e-1 1e-1];   % six videos
demo_scheme = {'none','1_2','ir',  'none','1_2','ir'};

%% 1) HOUSEKEEPING ---------------------------------------------------------
mkdir_if_missing(fullfile(PROJECT_ROOT, 'plots'));
mkdir_if_missing(fullfile(PROJECT_ROOT, 'videos'));

%% 2) READ & CACHE VIDEO ---------------------------------------------------
fprintf('Reading video ...\n');
vidObj         = VideoReader(aviFile);
H = vidObj.Height; W = vidObj.Width;
framesOriginal = read(vidObj);                     % ALL frames
nFrames        = size(framesOriginal, 4);

fprintf('Packetising %d frames …\n', nFrames);
packetsPerFrame = cell(1, nFrames);
for iFr = 1:nFrames
    bits = frame2bits(framesOriginal(:,:,:,iFr));
    numPk = floor(numel(bits) / infoLen);
    packetsPerFrame{iFr} = reshape(bits(1:numPk*infoLen), infoLen, numPk);
    fprintf('  • Frame %3d/%d → %d packets\n', iFr, nFrames, numPk);
end
totalInfoBits = sum(cellfun(@numel, packetsPerFrame));

%% 3) METRIC SWEEP ---------------------------------------------------------
trellis   = poly2trellis(7, motherRate);
ber_half  = zeros(size(p_vals));
ber_ir    = zeros(size(p_vals));
thr_ir    = zeros(size(p_vals));

fprintf('\n*** Metric sweep (%d points) ***\n', numel(p_vals));
ticSweep = tic;
for ip = 1:numel(p_vals)
    p = p_vals(ip);

    errH = 0; errIR = 0;  txH = 0; txIR = 0;
    for iFr = 1:nFrames
        pkMat = packetsPerFrame{iFr};
        for pk = 1:size(pkMat,2)
            infoBits = pkMat(:,pk);

            % baseline 1/2
            [dH, tH]  = transmit_once(infoBits, trellis, p, punct1_2, traceback);
            errH = errH + sum(dH ~= infoBits);  txH  = txH  + tH;

            % incremental redundancy
            [dIR, tIR] = transmit_IR(infoBits, trellis, p, rateTable, traceback);
            errIR = errIR + sum(dIR ~= infoBits); txIR = txIR + tIR;
        end
    end
    ber_half(ip) = errH  / totalInfoBits;
    ber_ir(ip)   = errIR / totalInfoBits;
    thr_ir(ip)   = totalInfoBits / txIR;

    % ETA print
    elapsed = toc(ticSweep);
    fprintf('[Sweep] p=%.4g  BER½=%.3e  BERir=%.3e  Thr=%.3f  ETA %.1fs\n', ...
            p, ber_half(ip), ber_ir(ip), thr_ir(ip), ...
            elapsed/ip*(numel(p_vals)-ip));
end

%% 4) PLOTS ---------------------------------------------------------------
figure;
semilogx(p_vals, ber_half,'o-', p_vals, ber_ir,'s-','LineWidth',1.3);
grid on; xlabel('Channel error probability p'); ylabel('BER');
title('Coded BER vs p'); legend('Rate 1/2','Incremental redundancy','Location','NW');
saveas(gcf, fullfile(PROJECT_ROOT, 'plots', 'BER_plot.png'));

figure;
semilogx(p_vals, thr_ir,'^-','LineWidth',1.3); grid on;
xlabel('Channel error probability p');
ylabel('Throughput  (useful bits / sent bits)');
title('Throughput vs p  (incremental redundancy)');
saveas(gcf, fullfile(PROJECT_ROOT, 'plots','Throughput_plot.png'));

%% 5) SHOWCASE DECODING (6 videos) ---------------------------------------
fprintf('\n*** Generating showcase videos ***\n');
for k = 1:6
    pShow  = demo_p(k);
    scheme = demo_scheme{k};
    outVid = fullfile(PROJECT_ROOT, 'videos', sprintf('decoded_%s_p%.0e.avi', scheme, pShow));

    fprintf('→ %s (%d/%d)\n', outVid, k, 6);
    decoded = decode_video_cached(packetsPerFrame, framesOriginal, ...
                                  pShow, scheme, trellis, traceback, ...
                                  rateTable, rateName, infoLen, H, W);
    write_video(decoded, outVid, vidObj.FrameRate);
end

fprintf('\nAll tasks complete – plots saved in /plots, videos in /videos.\n');

%% =======================  LOCAL FUNCTIONS  ===============================
function mkdir_if_missing(d); if ~isfolder(d), mkdir(d); end; end

function bits = frame2bits(rgb)
bits = reshape([de2bi(double(rgb(:,:,1))); de2bi(double(rgb(:,:,2))); ...
                de2bi(double(rgb(:,:,3)))], [], 1);
bits = logical(bits);
end

function rgb = bits2frame(bits,H,W)
N = H*W; mat = reshape(bits, [], 8);
r = uint8(bi2de(mat(1:N,:)));
g = uint8(bi2de(mat(N+1:2*N,:)));
b = uint8(bi2de(mat(2*N+1:end,:)));
rgb = cat(3, reshape(r,H,W), reshape(g,H,W), reshape(b,H,W));
end

function [d,tx] = transmit_once(info, trel, p, punc, tb)
enc = convenc(info, trel, punc); tx = numel(enc);
d   = vitdec(bsc(enc,p), trel, tb,'trunc','hard', punc);
end

function [d,tx] = transmit_IR(info, trel, p, rateTab, tb)
tx = 0;
for r = 1:numel(rateTab)
    punc = rateTab{r};
    enc  = convenc(info, trel, punc);  tx = tx + numel(enc);
    d    = vitdec(bsc(enc,p), trel, tb,'trunc','hard', punc);
    if all(d==info), return, end
end
end

function [d,tx] = transmit_uncoded(info, p)
d  = bsc(info, p);          % plain BSC
tx = numel(info);           % sent bits = info bits
end

function [d,tx] = transmit_scheme(info, trel, p, tb, scheme, rateTab, rateName)
switch scheme
    case 'none', [d,tx] = transmit_uncoded(info,p);
    case '1_2',  punc=rateTab{strcmp('1_2',rateName)};
                 [d,tx] = transmit_once(info,trel,p,punc,tb);
    case 'ir',   [d,tx] = transmit_IR(info,trel,p,rateTab,tb);
    otherwise    error('Unknown scheme %s',scheme);
end
end

function vid = decode_video_cached(pktCell, framesIn, p, scheme, ...
                                   trel, tb, rateTab, rateName, ...
                                   infoLen, H, W)
nF = numel(pktCell); vid = zeros(size(framesIn),'uint8');
for f = 1:nF
    pkMat = pktCell{f}; rec = false(infoLen*size(pkMat,2),1);
    for pk = 1:size(pkMat,2)
        [dec,~] = transmit_scheme(pkMat(:,pk), trel, p, tb, ...
                                   scheme, rateTab, rateName);
        rec((pk-1)*infoLen+(1:infoLen)) = dec;
    end
    tail = frame2bits(framesIn(:,:,:,f)); tail = tail(numel(rec)+1:end);
    vid(:,:,:,f) = bits2frame([rec;tail],H,W);
end
end

function write_video(frames, path, fps)
vw = VideoWriter(path,'Uncompressed AVI'); vw.FrameRate = fps;
open(vw); writeVideo(vw,frames); close(vw);
end

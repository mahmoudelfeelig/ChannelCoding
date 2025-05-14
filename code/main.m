%% Channel Coding Project – incremental redundancy
% COMM B504 – Spring 2025
% ------------------------------------------------------------
%  Author(s): Mahmoud Elfil, Omar Emad – 14‑May‑2025

run(fullfile('..','startup.m'));
clear; clc; close all;

%% 0) USER SETTINGS --------------------------------------------------------
aviFile        = fullfile('data','1.avi');      % original video
p_vals         = logspace(log10(1e-4),log10(2e-1),15);% BSC error probs
traceback      = 35;                                 % Viterbi traceback
infoLen        = 1024;                               % bits per packet
motherRate     = [171 133];                          % octal generators

% Puncturing patterns (1 = send, 0 = punctured)
punct8_9 = [1 1 1 1  0 1 1 1  1 0 0 0  1 0 0 0];   % 8/9
punct4_5 = [1 1 1 1  1 1 1 1  1 0 0 0  1 0 0 0];   % 4/5
punct2_3 = [1 1 1 1  1 1 1 1  1 0 1 0  1 0 1 0];   % 2/3
punct1_2 = ones(1,2);                              % 1/2 (no puncturing)

rateTable = {punct8_9,punct4_5,punct2_3,punct1_2};
rateName  = {'8_9','4_5','2_3','1_2'};             % safe for filenames

% Extra video demo settings (pairs ⟨p , scheme⟩ to export)
demo_p      = [1e-3  , 1e-1];                      % 0.001 & 0.1
demo_scheme = {'1_2','ir','8_9'};                  % 3 schemes  → 6 videos

%% 1) HOUSE‑KEEPING --------------------------------------------------------
mkdir_if_missing('plots');
mkdir_if_missing('videos');

%% 2) READ ORIGINAL VIDEO --------------------------------------------------
fprintf('Reading %s ...\n',aviFile);
vidObj          = VideoReader(aviFile);
H               = vidObj.Height;         % frame height
W               = vidObj.Width;          % frame width
framesOriginal  = read(vidObj);          % 4‑D uint8 array
nFrames         = size(framesOriginal,4);

%% 3) METRIC SIMULATION LOOP ----------------------------------------------
trellis = poly2trellis(7,motherRate);

ber_half = zeros(size(p_vals));          % baseline (rate‑1/2)
ber_ir   = zeros(size(p_vals));          % incremental redundancy
thr_ir   = zeros(size(p_vals));          % throughput (info / sent)

fprintf('\n*** Metric sweep (%d channel points) ***\n',numel(p_vals));
for ip = 1:numel(p_vals)
    p = p_vals(ip);
    fprintf(' p = %.4g ... ',p);
    
    % Counters ------------------------------------------------------------
    errBits_half = 0; errBits_ir = 0;
    txBits_half  = 0; txBits_ir  = 0;
    
    % Loop over every frame ----------------------------------------------
    for iFr = 1:nFrames
        binStream = frame2bits(framesOriginal(:,:,:,iFr));   % helper
        
        % Loop over packets ----------------------------------------------
        numPackets = floor(numel(binStream)/infoLen);
        for pk = 1:numPackets
            idx       = (pk-1)*infoLen + (1:infoLen);
            infoBits  = binStream(idx).';
            
            % (A) Baseline – single shot rate‑1/2 ------------------------
            [dec_half,tx_half] = transmit_once(infoBits,trellis,p,...
                                               punct1_2,traceback);
            errBits_half = errBits_half + sum(dec_half~=infoBits);
            txBits_half  = txBits_half  + tx_half;
            
            % (B) Incremental‑redundancy ---------------------------------
            [dec_ir,tx_ir]  = transmit_IR(infoBits,trellis,p,...
                                          rateTable,traceback);
            errBits_ir = errBits_ir + sum(dec_ir~=infoBits);
            txBits_ir  = txBits_ir  + tx_ir;
        end
    end
    
    % Final metrics for this p -------------------------------------------
    totalInfoBits = double(numPackets)*double(nFrames)*infoLen;
    ber_half(ip)  = errBits_half / totalInfoBits;
    ber_ir(ip)    = errBits_ir   / totalInfoBits;
    thr_ir(ip)    = totalInfoBits / txBits_ir;
    fprintf("done\n");
end

%% 4) PLOT & SAVE METRICS --------------------------------------------------
figure;
semilogx(p_vals,ber_half,'o-','LineWidth',1.3); hold on;
semilogx(p_vals,ber_ir,'s-','LineWidth',1.3); grid on;
xlabel('Channel error probability  p');
ylabel('Bit‑error rate (BER)'); title('BER vs p');
legend('Rate 1/2 baseline','Incremental redundancy','Location','northwest');
saveas(gcf,fullfile('plots','BER_plot.png'));

figure;
semilogx(p_vals,thr_ir,'^-','LineWidth',1.3); grid on;
xlabel('Channel error probability  p');
ylabel('Throughput  (useful bits / total bits)');
title('Throughput vs p  –  incremental redundancy');
saveas(gcf,fullfile('plots','Throughput_plot.png'));

%% 5) SHOWCASE VIDEOS (6 decodings) ---------------------------------------
fprintf('\n*** Producing showcase decoded videos ***\n');
for ip = 1:numel(demo_p)
    for is = 1:numel(demo_scheme)
        pShow = demo_p(ip);
        scheme = demo_scheme{is};       % '1_2' | 'ir' | '8_9' | ...
        outName = sprintf('decoded_p%.0e_%s.avi',pShow,scheme);
        outName = fullfile('videos',outName);
        
        fprintf('  → %s ... ',outName);
        decodedFrames = decode_video(framesOriginal,pShow,scheme,...
                                     trellis,traceback,...
                                     rateTable,rateName,infoLen);
        write_video(decodedFrames,outName,vidObj.FrameRate);
        fprintf("saved\n");
    end
end

fprintf('\nAll tasks completed successfully.\n');

%% =======================  LOCAL FUNCTIONS  ==============================
%  (Kept at end of script for clarity.  Move to separate .m files if desired)

% -------------------------------------------------------------------------
function mkdir_if_missing(dirname)
% create directory if it does not exist
if ~exist(dirname,'dir'), mkdir(dirname); end
end

% -------------------------------------------------------------------------
function bin = frame2bits(frameRGB)
% Convert H×W×3 uint8 frame to a column vector of bits (LSB first).
R = frameRGB(:,:,1); G = frameRGB(:,:,2); B = frameRGB(:,:,3);
bin = reshape([de2bi(double(R(:))); ...
               de2bi(double(G(:))); ...
               de2bi(double(B(:)))],[],1);
end

% -------------------------------------------------------------------------
function frameRGB = bits2frame(bin,H,W)
% Inverse of frame2bits.
N      = H*W;
bitMat = reshape(bin,[],8);         % (#rows = 3N,  8 bits each)
Rval   = uint8(bi2de(bitMat(1:N       ,:)));
Gval   = uint8(bi2de(bitMat(N+1:2*N ,:)));
Bval   = uint8(bi2de(bitMat(2*N+1:end,:)));
frameRGB = cat(3,reshape(Rval,H,W), ...
                  reshape(Gval,H,W), ...
                  reshape(Bval,H,W));
end

% -------------------------------------------------------------------------
function [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb)
% Single‑shot transmission with fixed puncturing pattern.
enc      = convenc(infoBits,trellis,punc);
txBits   = numel(enc);
rx       = bsc(enc,p);                    % Binary Symmetric Channel
decBits  = vitdec(rx,trellis,tb,'trunc','hard',punc);
end

% -------------------------------------------------------------------------
function [decBits,txBits] = transmit_IR(infoBits,trellis,p,rateTab,tb)
% Incremental‑redundancy HARQ as per project brief.
txBits = 0;
for rr = 1:numel(rateTab)
    punc     = rateTab{rr};
    encPart  = convenc(infoBits,trellis,punc);
    txBits   = txBits + numel(encPart);
    rxPart   = bsc(encPart,p);
    decBits  = vitdec(rxPart,trellis,tb,'trunc','hard',punc);
    if all(decBits==infoBits), return, end          % success
end
% If we fall through, decBits already holds final attempt (1/2).
end

% -------------------------------------------------------------------------
function [decBits,txBits] = transmit_scheme(infoBits,trellis,p,tb,...
                                            scheme,rateTab,rateName)
% Wraps all supported schemes: '1_2','ir','8_9','4_5','2_3'
switch scheme
    case {'1_2','half'}
        idx      = strcmp('1_2',rateName);
        punc     = rateTab{idx};
        [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb);
        
    case 'ir'
        [decBits,txBits] = transmit_IR(infoBits,trellis,p,rateTab,tb);
        
    case {'8_9','4_5','2_3'}
        idx      = strcmp(scheme,rateName);
        punc     = rateTab{idx};
        [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb);
        
    otherwise
        error('Unknown scheme "%s"',scheme);
end
end

% -------------------------------------------------------------------------
function decodedFrames = decode_video(framesIn,p,scheme, ...
                                      trellis,tb, ...
                                      rateTab,rateName,infoLen)
% Decode *all* frames of a video under a given channel condition & scheme.
[H,W,~,nF] = size(framesIn);
decodedFrames = zeros(size(framesIn),'uint8');

wb = waitbar(0,'Decoding video ...');         % progress bar
cleanWB = onCleanup(@() delete(wb));

for iFr = 1:nF
    waitbar(iFr/nF,wb,sprintf('Decoding frame %d / %d',iFr,nF));
    
    binStream = frame2bits(framesIn(:,:,:,iFr));
    numPackets = floor(numel(binStream)/infoLen);
    
    % Pre‑allocate recovered bitstream
    recStream = zeros(numPackets*infoLen,1);
    
    for pk = 1:numPackets
        idx      = (pk-1)*infoLen + (1:infoLen);
        infoBits = binStream(idx).';
        [decBits,~] = transmit_scheme(infoBits,trellis,p,tb,...
                                      scheme,rateTab,rateName);
        recStream(idx) = decBits;
    end
    
    % Put back the remainder bits (if any) untouched for perfect length
    tailIdx = numPackets*infoLen+1 : numel(binStream);
    recStream = [recStream ; binStream(tailIdx)]; %#ok<AGROW>
    
    decodedFrames(:,:,:,iFr) = bits2frame(recStream,H,W);
end
end

% -------------------------------------------------------------------------
function write_video(frames,outFile,fps)
% Save a 4‑D uint8 array of frames to disk.
vw = VideoWriter(outFile,'Uncompressed AVI');
vw.FrameRate = fps;
open(vw);
writeVideo(vw,frames);
close(vw);
end

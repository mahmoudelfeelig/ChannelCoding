%% Channel Coding Project – incremental redundancy
% COMM B504 – Spring 2025
% ------------------------------------------------------------
%  Author(s): Mahmoud Elfil, Omar Emad – 14‑May‑2025

run(fullfile('..','startup.m'));      % ensure paths
clear; clc; close all;

%% 0) USER SETTINGS --------------------------------------------------------
aviFile        = fullfile('data','2.avi');         % original video
p_vals         = logspace(log10(1e-4),log10(2e-1),15);
p_vals = p_vals(1:5); % REMOVE TS AFTER YOURE DONE TESTING AAAAAAAAAAAAAAAAAA
traceback      = 35;                               % Viterbi tb length
infoLen        = 1024;                             % bits per packet
motherRate     = [171 133];                        % octal generators

% Puncturing patterns
punct8_9 = [1 1 1 1  0 1 1 1  1 0 0 0  1 0 0 0];
punct4_5 = [1 1 1 1  1 1 1 1  1 0 0 0  1 0 0 0];
punct2_3 = [1 1 1 1  1 1 1 1  1 0 1 0  1 0 1 0];
punct1_2 = ones(1,2);

rateTable = {punct8_9,punct4_5,punct2_3,punct1_2};
rateName  = {'8_9','4_5','2_3','1_2'};

% Showcase video settings
demo_p      = [1e-3  , 1e-1];
demo_scheme = {'1_2','ir','8_9'};

%% 1) HOUSE‑KEEPING --------------------------------------------------------
mkdir_if_missing('plots'); mkdir_if_missing('videos');

%% 2) READ & CACHE VIDEO ---------------------------------------------------
fprintf('Reading %s ...\n',aviFile);
vidObj          = VideoReader(aviFile);
H               = vidObj.Height;  W = vidObj.Width;
framesOriginal  = read(vidObj);
maxFrames      = 5;
nFrames        = min(maxFrames, size(framesOriginal, 4));
nFrames         = size(framesOriginal,4);

fprintf('Packetising frames (one‑off)... ');
packetsPerFrame = cell(1,nFrames);   % each cell: infoLen × numPackets logical
for iFr = 1:nFrames
    bits = frame2bits(framesOriginal(:,:,:,iFr));
    numPackets = floor(numel(bits)/infoLen);
    packetsPerFrame{iFr} = reshape(bits(1:numPackets*infoLen), ...
                                   infoLen, numPackets);
end
fprintf("done (cached in RAM).\n");

totalInfoBits = sum(cellfun(@numel, packetsPerFrame)); % for throughput later

%% 3) METRIC SWEEP (PARALLEL) ---------------------------------------------
trellis = poly2trellis(7,motherRate);
numP    = numel(p_vals);
ber_half = zeros(1,numP);  ber_ir = zeros(1,numP);  thr_ir = zeros(1,numP);

fprintf('\n*** Metric sweep in parallel (%d points) ***\n',numP);

% Attempt to start a parallel pool (gracefully fallback if toolbox missing)
try
    pool = gcp('nocreate');
    if isempty(pool), pool = parpool; end
catch
    warning('Parallel Computing Toolbox not available – running serial.');
end

parfor (ip = 1:numP, maxNumCompThreads)   % cores automatically selected
    p = p_vals(ip);
    
    err_half = 0;  err_ir = 0;
    tx_half  = 0;  tx_ir  = 0;
    
    for iFr = 1:nFrames
        pkMat = packetsPerFrame{iFr};     % infoLen × numPackets
        numPackets = size(pkMat,2);
        
        for pk = 1:numPackets
            infoBits = pkMat(:,pk);
            
            % baseline 1/2
            [decH,txH] = transmit_once(infoBits,trellis,p,punct1_2,traceback);
            err_half = err_half + sum(decH~=infoBits);
            tx_half  = tx_half  + txH;
            
            % IR scheme
            [decIR,txIR] = transmit_IR(infoBits,trellis,p,rateTable,traceback);
            err_ir = err_ir + sum(decIR~=infoBits);
            tx_ir  = tx_ir  + txIR;
        end
    end
    
    ber_half(ip) = err_half / totalInfoBits;
    ber_ir(ip)   = err_ir   / totalInfoBits;
    thr_ir(ip)   = totalInfoBits / tx_ir;
    
    fprintf('  [worker %d] p = %.4g  →  BER_ir = %.3e, Thr = %.3f\n', ...
             getCurrentTaskID, p, ber_ir(ip), thr_ir(ip));
end

%% 4) PLOT & SAVE METRICS --------------------------------------------------
figure; semilogx(p_vals,ber_half,'o-',p_vals,ber_ir,'s-','LineWidth',1.3);
grid on; xlabel('Channel error probability  p'); ylabel('BER');
title('BER vs p'); legend('Rate 1/2','Incremental redundancy','Location','NW');
saveas(gcf,fullfile('plots','BER_plot.png'));

figure; semilogx(p_vals,thr_ir,'^-','LineWidth',1.3); grid on;
xlabel('Channel error probability  p');
ylabel('Throughput  (useful bits / total bits)');
title('Throughput vs p – incremental redundancy');
saveas(gcf,fullfile('plots','Throughput_plot.png'));

%% 5) SHOWCASE VIDEOS ------------------------------------------------------
fprintf('\n*** Producing showcase decoded videos ***\n');
for ip = 1:numel(demo_p)
    for is = 1:numel(demo_scheme)
        pShow  = demo_p(ip);
        scheme = demo_scheme{is};
        outVid = sprintf('decoded_p%.0e_%s.avi',pShow,scheme);
        outVid = fullfile('videos',outVid);
        
        fprintf('  → %s ... ',outVid);
        decFrames = decode_video_cached(packetsPerFrame,framesOriginal,...
                                        pShow,scheme,trellis,traceback,...
                                        rateTable,rateName,infoLen,H,W);
        write_video(decFrames,outVid,vidObj.FrameRate);
        fprintf("saved\n");
    end
end

fprintf('\nAll tasks completed successfully.\n');

%% =======================  LOCAL FUNCTIONS  ==============================
function id = getCurrentTaskID
% returns a small integer identifying the parfor worker (or 0 in serial)
t = getCurrentTask;
if isempty(t), id = 0; else, id = t.ID; end
end

function mkdir_if_missing(d)
if ~isfolder(d), mkdir(d); end
end

function bin = frame2bits(frameRGB)
R = frameRGB(:,:,1); G = frameRGB(:,:,2); B = frameRGB(:,:,3);
bin = reshape([de2bi(double(R(:))); de2bi(double(G(:))); de2bi(double(B(:)))],[],1);
bin = logical(bin);              % store as logical to save 8× space
end

function frameRGB = bits2frame(bin,H,W)
N = H*W;
bitMat = reshape(bin,[],8);    % (#rows = 3N)
R = uint8(bi2de(bitMat(1:N      ,:)));
G = uint8(bi2de(bitMat(N+1:2*N ,:)));
B = uint8(bi2de(bitMat(2*N+1:end,:)));
frameRGB = cat(3,reshape(R,H,W),reshape(G,H,W),reshape(B,H,W));
end

function [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb)
enc = convenc(infoBits,trellis,punc);
txBits = numel(enc);
decBits = vitdec(bsc(enc,p),trellis,tb,'trunc','hard',punc);
end

function [decBits,txBits] = transmit_IR(infoBits,trellis,p,rateTab,tb)
txBits = 0;
for r = 1:numel(rateTab)
    punc = rateTab{r};
    enc  = convenc(infoBits,trellis,punc);
    txBits = txBits + numel(enc);
    decBits = vitdec(bsc(enc,p),trellis,tb,'trunc','hard',punc);
    if all(decBits==infoBits), return; end
end
end

function decFrames = decode_video_cached(pktCell,framesIn,p,scheme,...
                                         trellis,tb,rateTab,rateName,...
                                         infoLen,H,W)
nF = numel(pktCell);
decFrames = zeros(size(framesIn),'uint8');
wb = waitbar(0,'Decoding showcase video...');
clean = onCleanup(@() delete(wb));

for iFr = 1:nF
    pkMat = pktCell{iFr};
    numPackets = size(pkMat,2);
    recStream  = false(infoLen*numPackets,1);
    
    for pk = 1:numPackets
        infoBits = pkMat(:,pk);
        [decBits,~] = transmit_scheme(infoBits,trellis,p,tb,scheme,rateTab,rateName);
        recStream((pk-1)*infoLen + (1:infoLen)) = decBits;
    end
    
    % Append any remainder bits untouched
    frameBits = frame2bits(framesIn(:,:,:,iFr));
    fullBits = [recStream ; frameBits((numPackets*infoLen+1):end)];
    decFrames(:,:,:,iFr) = bits2frame(fullBits,H,W);
    
    waitbar(iFr/nF,wb);
end
end

function [decBits,txBits] = transmit_scheme(infoBits,trellis,p,tb,...
                                            scheme,rateTab,rateName)
switch scheme
    case {'1_2','half'}
        punc = rateTab{strcmp('1_2',rateName)};
        [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb);
    case 'ir'
        [decBits,txBits] = transmit_IR(infoBits,trellis,p,rateTab,tb);
    otherwise   % 8_9, 4_5, 2_3
        punc = rateTab{strcmp(scheme,rateName)};
        [decBits,txBits] = transmit_once(infoBits,trellis,p,punc,tb);
end
end

function write_video(frames,outFile,fps)
vw = VideoWriter(outFile,'Uncompressed AVI');
vw.FrameRate = fps; open(vw); writeVideo(vw,frames); close(vw);
end

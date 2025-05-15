% === startup.m ===
% Automatically adds the current project folder (and subfolders) to MATLAB path
% and sets working directory to the /code subfolder if it exists.

fprintf('[startup] Initializing project environment...\n');

% Get full path of the startup script
projectRoot = fileparts(mfilename('fullpath'));

% Add project folders to path
addpath(genpath(projectRoot));
fprintf('[startup] Added project and subfolders to MATLAB path.\n');

% Save to base workspace so main.m can use it
assignin('base', 'PROJECT_ROOT', projectRoot);

% Change directory to /code if it exists
codeDir = fullfile(projectRoot, 'code');
if isfolder(codeDir)
    cd(codeDir);
    fprintf('[startup] Changed working directory to /code.\n');
else
    warning('[startup] /code folder not found. Staying in %s.\n', projectRoot);
end
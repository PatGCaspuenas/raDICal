
function InterpolateDatasets(Nimg,X,Y,RootFields,RootGrid,RootOut,FlagView)

%%% Inputs %%%
%
% Nimg          -> Snapshots used (from Flow1 to Flow 10)
% RootFields    -> string of the type '...\Flow.' (location of file)
% RootGrid      -> same as RootFields + Nimg (just for one)
% Rootout       -> filename to save the obtained data
% FlagView      -> 'YES' or any other thing
% X             -> My X domain
% Y             -> My Y domain


Grid=OpenGrid(RootGrid);

Utest=zeros(numel(X),numel(Nimg));
Vtest=zeros(numel(X),numel(Nimg));
Ptest=Vtest;

for i=1:numel(Nimg)
    disp(i)
    s=sprintf('%s%06d',RootFields,Nimg(i));
    Flow=OpenFields(s);
    F=scatteredInterpolant(Grid.x,Grid.y,Flow.u,'natural');
    U=F(X,Y);
    F=scatteredInterpolant(Grid.x,Grid.y,Flow.v,'natural');
    V=F(X,Y);
    F=scatteredInterpolant(Grid.x,Grid.y,Flow.p,'natural');
    P=F(X,Y);

    u(:,i)=U(:);
    v(:,i)=V(:);
    p(:,i)=P(:);

    if strcmp(FlagView,'YES')
        figure(1)
        pcolor(X,Y,P)
        shading interp
        colormap jet(16)
        axis equal
        hold on
        fill(-(3/2)*cosd(30)+0.5*cos(0:0.1*pi:2*pi),0+0.5*sin(0:0.1*pi:2*pi),'w');
        fill(0+0.5*cos(0:0.1*pi:2*pi),-(3/4)+0.5*sin(0:0.1*pi:2*pi),'w');
        fill(0+0.5*cos(0:0.1*pi:2*pi),(3/4)+0.5*sin(0:0.1*pi:2*pi),'w');
        pause(0.1)
    end    
end

save(sprintf('%s',RootOut),'u','v','p','Nimg','X','Y')
end

function Flow=OpenFields(filename)

%% Initialize variables.
% filename = 'F:\Pinball\Dataset_Re130\Code_Output\Flow.014901';

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%31f%30f%f%[^\n\r]'; %[..] is a line break 

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);

% Cell array

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

%% Create output variable
Flow = table(dataArray{1:end-1}, 'VariableNames', {'u','v','p'});

%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;
end

function Grid=OpenGrid(filename)

%% Initialize variables.
% filename = 'F:\Pinball\Dataset_Re130\Code_Output\Flow.014901';

%% Format for each line of text:
%   column1: double (%f)
%	column2: double (%f)
%   column3: double (%f)
% For more information, see the TEXTSCAN documentation.
formatSpec = '%31f%30f%f%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, 'Delimiter', '', 'WhiteSpace', '', 'TextType', 'string', 'EmptyValue', NaN,  'ReturnOnError', false);

%% Close the text file.
fclose(fileID);

%% Post processing for unimportable data.
% No unimportable data rules were applied during the import, so no post
% processing code is included. To generate code which works for
% unimportable data, select unimportable cells in a file and regenerate the
% script.

NumPoints=dataArray{1}(1);
dataArray{3}=[];
dataArray{4}=[];
for i=1:2
dataArray{i}(1)=[];
dataArray{i}(NumPoints+1:end)=[];
end

%% Create output variable
Grid = table(dataArray{1:2}, 'VariableNames', {'x','y'});

%% Clear temporary variables
clearvars filename formatSpec fileID dataArray ans;
end
annPath ='../data/annForDb_otb.mat';
setFd = '/disk2/zfchen/data/OTB_sentences/OTB_videos';
outFd = '../data/otbEbRp';
addpath('../fun');
addpath('../WSRef/edges');
addpath(genpath('/mnt/lustre/chenzhenfang/code/WSSPL/WSRef/toolbox'));  savepath;

%% load pre-trained edge detection model and set opts (see edgesDemo.m)
model=load('../WSRef/edges/models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
maxRPNum =1000;

ann=load(annPath);
% test set
testVdNum = size(ann.testImg, 2);
for i=1:testVdNum
    %continue;
    vdName = strtrim(ann.testName(i, :));
    tmpFrmList = ann.testImg{i};
    vdFrmNum = size(tmpFrmList, 1);
    outFdSub = [outFd '/' vdName];
    mkdir(outFdSub);
    inFdSub = [setFd '/' vdName];

    for j=1:vdFrmNum
        tmpImg = [inFdSub '/img/' tmpFrmList(j, :) '.jpg'];
        outFn = [outFdSub '/' tmpFrmList(j, :) '.mat'];
        if(exist(outFn, 'file'))
            continue;
        end
        fprintf(' extracting proposals for the %d frame of %s\n', j, vdName)
        I = imread(tmpImg);
        [h, w, c] = size(I);
        if(c==1)
            I = cat(3, I, I, I);
        end
        tic, bbs=edgeBoxes(I,model,opts); toc
        if(size(bbs, 1)>maxRPNum)
            bbs = bbs(1:maxRPNum, :);
        end
        save(outFn, 'bbs');
        
    end
end

    

trainVdNum = size(ann.trainImg, 2);
for i=1:trainVdNum
    vdName = strtrim(ann.trainName(i, :));
    tmpFrmList = ann.trainImg{i};
    vdFrmNum = size(tmpFrmList, 1);
    outFdSub = [outFd '/' vdName];
    mkdir(outFdSub);
    inFdSub = [setFd '/' vdName];
    for j=1:vdFrmNum
        tmpImg = [inFdSub '/img/' tmpFrmList(j, :) '.jpg'];
        outFn = [outFdSub '/' tmpFrmList(j, :) '.mat'];
        if(exist(outFn, 'file'))
            continue;
        end
        fprintf(' extracting proposals for the %d frame of %s\n', j, vdName)
        I = imread(tmpImg);
        [h, w, c] = size(I);
        if(c==1)
            I = cat(3, I, I, I);
        end
        tic, bbs=edgeBoxes(I,model,opts); toc
        if(size(bbs, 1)>maxRPNum)
            bbs = bbs(1:maxRPNum, :);
        end
        save(outFn, 'bbs');
    end
end


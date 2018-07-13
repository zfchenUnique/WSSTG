python ../fun/create_word2vec_for_dataset.py --dictOutPath ../data/dictForDb \
    --setOutPath  ../data/annoted  \
    --setName a2d \
    --annFn /disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_annotation.txt \
    --annFd /disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_annotation_with_instances \
    --annIgListFn /disk2/zfchen/data/A2D/Release/sentenceAnno/a2d_missed_videos.txt \
    --annOriFn /disk2/zfchen/data/A2D/Release/videoset.csv

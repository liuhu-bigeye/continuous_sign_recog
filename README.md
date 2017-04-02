0. end to end model
    all vgg s from triplet pretrain
    resample, img augmentation, conv1d dropout 0.1
    best epoch: 63, dev/test: 0.412, 0.411
    model file: all_vgg_s/output/ctc_2017-03-23_13-25-15/best_epochs_63_0.412_0.411/ctc_all_vgg_s_epoch_0063

1. alignment making
    hard ctc aligmnet, weighted categorical cross entrophy
    filtered ctc loss > 5 or best path loss > 5
    filtered offset > 0, due to cut of gt and consequently large ctc loss when offset == 3
    alignment file: all_vgg_s/output/ctc_predict_2017-03-28_20-41-05/ctc_best_path_63_0.412_0.411_off0.pkl

2. cnn training
    from triplet pretrain
    img augmentation, conv1d dropout 0.5
    please follow log.txt under cnn_training/output/cnn_training_2017-03-30_16-04-22


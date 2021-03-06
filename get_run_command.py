# standard config :
cfg = {
    "MODEL": "LinearModelWeightedDrop",
    "MODEL_DIR": "LinearModel",
    "DATASET_PROCESSOR": "MNISTLoader",
    "OUTPUT_TYPE": "NUMBERS",
    "COLUMNS_TYPE": "OHV_ALL",
    "LOAD_METHOD": "OVERSAMPLE_NON_MAX_TRAIN",
    "LOSS": "NLLLoss",
    "DROP_PROB": 0.2,
    "OPTIMIZER": "SGD",
    "LR": 0.001,
    "BATCH_SIZE": 32,
    "EPOCH": 3,
    "PRINT_EVERY": 10,
    "EVALUATION_METHODS": [
        "ACCURACY",
        "CONFUSION_MATRIX",
        "PRECISION",
        "RECALL",
        "F1_SCORE",
        "GMEAN"
    ],
    "PROB_METHOD": "TANH"
}

# All combinations of differet dataset, models and split:
changes = [
    # ["bow_nd_binary.json", "LinearModelNoDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_sd_binary.json", "LinearModelDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_standout_binary.json", "LinearModelStandout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_gd_tanh_binary.json", "LinearModelGradBasedDrop", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", None],

    #  ["bow_nd_grade.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_sd_grade.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_standout_grade.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["bow_gd_tanh_grade.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", None],

    #  ["bow_nd_binary_50_50.json", "LinearModelNoDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["bow_sd_binary_50_50.json", "LinearModelDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["bow_standout_binary_50_50.json", "LinearModelStandout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["bow_gd_tanh_binary_50_50.json", "LinearModelGradBasedDrop", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [500,500]],

    #  ["bow_nd_grade_20_all.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["bow_sd_grade_20_all.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["bow_standout_grade_20_all.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["bow_gd_tanh_grade_20_all.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [200,200,200,200,200]],

    #  ["bow_nd_binary_10_90.json", "LinearModelNoDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["bow_sd_binary_10_90.json", "LinearModelDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["bow_standout_binary_10_90.json", "LinearModelStandout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["bow_gd_tanh_binary_10_90.json", "LinearModelGradBasedDrop", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [100,900]],

    #  ["bow_nd_grade_80_5.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["bow_sd_grade_80_5.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["bow_standout_grade_80_5.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["bow_wd_high_grade_80_5.json", "LinearModelWeightedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["bow_wd_low_grade_80_5.json", "LinearModelWeightedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "NORM", [50,50,50,50,800]],
    #  ["bow_gd_norm_grade_80_5.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "NORM", [50,50,50,50,800]],
    #  ["bow_gd_tanh_grade_80_5.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [50,50,50,50,800]],

    #  ["bow_nd_binary_30_70.json", "LinearModelNoDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["bow_sd_binary_30_70.json", "LinearModelDropout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["bow_standout_binary_30_70.json", "LinearModelStandout", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["bow_gd_tanh_binary_30_70.json", "LinearModelGradBasedDrop", "LinearModel", "BCELoss",
    #  "BoWKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [300,700]],

    #  ["bow_nd_grade_60_10.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["bow_sd_grade_60_10.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["bow_standout_grade_60_10.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["bow_gd_tanh_grade_60_10.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "BoWKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [100,100,100,100,600]],

    #  ### MNIST
    #  ["mnist_nd.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["mnist_sd.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["mnist_standout.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["mnist_gd_tanh.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, False, "TANH", None],

    #  ["mnist_nd_20_all.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]],
    #  ["mnist_sd_20_all.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]],
    #  ["mnist_standout_20_all.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]],
    #  ["mnist_gd_tanh_20_all.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, False, "TANH", [5000,5000,5000,5000,5000,5000,5000,5000,5000,5000]],

    #  ["mnist_nd_80_2.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [40000,1111,1111,1111,1111,1111,1111,1111,1111,1112]],
    #  ["mnist_sd_80_2.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [40000,1111,1111,1111,1111,1111,1111,1111,1111,1112]],
    #  ["mnist_standout_80_2.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [40000,1111,1111,1111,1111,1111,1111,1111,1111,1112]],
    #  ["mnist_gd_tanh_80_2.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, False, "TANH", [40000,1111,1111,1111,1111,1111,1111,1111,1111,1112]],

    #  ["mnist_nd_60_4.json", "LinearModelNoDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [30000,2222,2222,2222,2222,2222,2222,2222,2222,2224]],
    #  ["mnist_sd_60_4.json", "LinearModelDropout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [30000,2222,2222,2222,2222,2222,2222,2222,2222,2224]],
    #  ["mnist_standout_60_4.json", "LinearModelStandout", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, True, "NORM", [30000,2222,2222,2222,2222,2222,2222,2222,2222,2224]]
    #  ["mnist_gd_tanh_60_4.json", "LinearModelGradBasedDrop", "LinearModel", "NLLLoss",
    #  "MNISTLoader", "NUMBERS", "ORIGINAL", 32, 1000, 10, False, "TANH", [30000,2222,2222,2222,2222,2222,2222,2222,2222,2224]],
    
    ## OHV

    # ["grit_nd_binary.json", "GritNetNoDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_sd_binary.json", "GritNetSingleDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_gd_tanh_binary.json", "GritNetPerStepGradBasedDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", None],
    #  ["grit_psd_binary.json", "GritNetPerStepDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_prnn_binary.json", "GritNetPerStepRNNDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_prd_binary.json", "GritNetPerStepRecurrentDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_pss_binary.json", "GritNetPerStepStandout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", None],

    #  ["grit_nd_grade.json", "GritNetNoDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_sd_grade.json", "GritNetSingleDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_gd_tanh_grade.json", "GritNetPerStepGradBasedDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", None],
    #  ["grit_psd_grade.json", "GritNetPerStepDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_prnn_grade.json", "GritNetPerStepRNNDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_prd_grade.json", "GritNetPerStepRecurrentDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],
    #  ["grit_pss_grade.json", "GritNetPerStepStandout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", None],

    #  ["grit_nd_binary_50_50.json", "GritNetNoDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["grit_sd_binary_50_50.json", "GritNetSingleDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["grit_gd_tanh_binary_50_50.json", "GritNetPerStepGradBasedDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [500,500]],
    #  ["grit_psd_binary_50_50.json", "GritNetPerStepDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["grit_prnn_binary_50_50.json", "GritNetPerStepRNNDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["grit_prd_50_50.json", "GritNetPerStepRecurrentDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],
    #  ["grit_pss_50_50.json", "GritNetPerStepStandout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [500,500]],

    #  ["grit_nd_grade_20_all.json", "GritNetNoDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["grit_sd_grade_20_all.json", "GritNetSingleDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["grit_gd_tanh_grade_20_all.json", "GritNetPerStepGradBasedDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [200,200,200,200,200]],
    #  ["grit_psd_grade_20_all.json", "GritNetPerStepDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [200,200,200,200,200]],
    #  ["grit_prnn_grade_20_all.json", "GritNetPerStepRNNDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "NORM", [200,200,200,200,200]],
    #  ["grit_prd_gade_20_all.json", "GritNetPerStepRecurrentDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "NORM", [200,200,200,200,200]],
    #  ["grit_pss_grade_20_all.json", "GritNetPerStepStandout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [200,200,200,200,200]],


    #  ["grit_nd_binary_10_90.json", "GritNetNoDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["grit_sd_binary_10_90.json", "GritNetSingleDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["grit_gd_tanh_binary_10_90.json", "GritNetPerStepGradBasedDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [100,900]],
    #  ["grit_psd__binary_10_90.json", "GritNetPerStepDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["grit_prnn_binary_10_90.json", "GritNetPerStepRNNDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["grit_prd_binary_10_90.json", "GritNetPerStepRecurrentDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],
    #  ["grit_pss_binary_10_90.json", "GritNetPerStepStandout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [100,900]],

    #  ["grit_nd_grade_80_5.json", "GritNetNoDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["grit_sd_grade_80_5.json", "GritNetSingleDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["grit_gd_tanh_grade_80_5.json", "GritNetPerStepGradBasedDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [50,50,50,50,800]],
    #  ["grit_psd_grade_80_5.json", "GritNetPerStepDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["grit_prnn_grade_80_5.json", "GritNetPerStepRNNDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["grit_prd_grade_80_5.json", "GritNetPerStepRecurrentDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],
    #  ["grit_pss_grade_80_5.json", "GritNetPerStepStandout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [50,50,50,50,800]],

    #  ["grit_nd_binary_30_70.json", "GritNetNoDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["grit_sd_binary_30_70.json", "GritNetSingleDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["grit_gd_tanh_binary_30_70.json", "GritNetPerStepGradBasedDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, False, "TANH", [300,700]],
    #  ["grit_psd_binary_30_70.json", "GritNetPerStepDropout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["grit_prnn_binary_30_70.json", "GritNetPerStepRNNDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["grit_prd_binary_30_70.json", "GritNetPerStepRecurrentDrop", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],
    #  ["grit_pss_binary_30_70.json", "GritNetPerStepStandout", "GritNet", "BCELoss",
    #  "OneHotKEATS", "BINARY", "ORIGINAL", 4, 10, 10, True, "NORM", [300,700]],

    #  ["grit_nd_grade_60_10.json", "GritNetNoDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["grit_sd_grade_60_10.json", "GritNetSingleDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["grit_gd_tanh_grade_60_10.json", "GritNetPerStepGradBasedDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, False, "TANH", [100,100,100,100,600]],
    #  ["grit_psd_grade_60_10.json", "GritNetPerStepDropout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["grit_prnn_grade_60_10.json", "GritNetPerStepRNNDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["grit_prd_grade_60_10.json", "GritNetPerStepRecurrentDrop", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],
    #  ["grit_pss_grade_60_10.json", "GritNetPerStepStandout", "GritNet", "NLLLoss",
    #  "OneHotKEATS", "GRADE", "ORIGINAL", 4, 10, 10, True, "NORM", [100,100,100,100,600]],

    #     ["sentiment_nd_binary.json", "SimpleSentimentNoDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["sentiment_sd_binary.json", "SimpleSentimentSingleDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["sentiment_gd_tanh_binary.json", "SimpleSentimentPerStepGradBasedDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, False, "TANH", None],
    #  ["sentiment_psd_binary.json", "SimpleSentimentPerStepDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["sentiment_prnn_binary.json", "SimpleSentimentPerStepRNNDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["sentiment_prd_binary.json", "SimpleSentimentPerStepRecurrentDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],
    #  ["sentiment_pss_binary.json", "SimpleSentimentPerStepStandout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", None],


    #  ["sentiment_nd_binary_50_50.json", "SimpleSentimentNoDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],
    #  ["sentiment_sd_binary_50_50.json", "SimpleSentimentSingleDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],
    #  ["sentiment_gd_tanh_binary_50_50.json", "SimpleSentimentPerStepGradBasedDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, False, "TANH", [250000,250000]],
    #  ["sentiment_psd_binary_50_50.json", "SimpleSentimentPerStepDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],
    #  ["sentiment_prnn_binary_50_50.json", "SimpleSentimentPerStepRNNDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],
    #  ["sentiment_prd_50_50.json", "SimpleSentimentPerStepRecurrentDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],
    #  ["sentiment_pss_50_50.json", "SimpleSentimentPerStepStandout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [250000,250000]],


    #  ["sentiment_nd_binary_10_90.json", "SimpleSentimentNoDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],
    #  ["sentiment_sd_binary_10_90.json", "SimpleSentimentSingleDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],
    #  ["sentiment_gd_tanh_binary_10_90.json", "SimpleSentimentPerStepGradBasedDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, False, "TANH", [50000,450000]],
    #  ["sentiment_psd__binary_10_90.json", "SimpleSentimentPerStepDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],
    #  ["sentiment_prnn_binary_10_90.json", "SimpleSentimentPerStepRNNDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],
    #  ["sentiment_prd_binary_10_90.json", "SimpleSentimentPerStepRecurrentDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],
    #  ["sentiment_pss_binary_10_90.json", "SimpleSentimentPerStepStandout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [50000,450000]],


    #  ["sentiment_nd_binary_30_70.json", "SimpleSentimentNoDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
    #  ["sentiment_sd_binary_30_70.json", "SimpleSentimentSingleDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
    #  ["sentiment_gd_tanh_binary_30_70.json", "SimpleSentimentPerStepGradBasedDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, False, "TANH", [150000,350000]],
    #  ["sentiment_psd_binary_30_70.json", "SimpleSentimentPerStepDropout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
    #  ["sentiment_prnn_binary_30_70.json", "SimpleSentimentPerStepRNNDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
    #  ["sentiment_prd_binary_30_70.json", "SimpleSentimentPerStepRecurrentDrop", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
    #  ["sentiment_pss_binary_30_70.json", "SimpleSentimentPerStepStandout", "SimpleSentiment", "BCELoss",
    #  "Sentiment140", "BINARY", "ORIGINAL", 32, 1000, 10, True, "NORM", [150000,350000]],
]

import json
import os

# iterate through all combination and write a json file for each
for each_change in changes:
    print("here")
    with open(".\\configs\\"+each_change[0], 'w') as f:
        c = cfg
        c['MODEL'] = each_change[1]
        c['MODEL_DIR'] = each_change[2]
        c['LOSS'] = each_change[3]
        c['DATASET_PROCESSOR'] = each_change[4]
        c['OUTPUT_TYPE'] = each_change[5]
        c['LOAD_METHOD'] = each_change[6]
        c['BATCH_SIZE'] = each_change[7]
        c['PRINT_EVERY'] = each_change[8]
        c['EPOCH'] = each_change[9]
        c['KEEP_HIGH_MAGNITUDE'] = each_change[10]
        c['PROB_METHOD'] = each_change[11]
        c['CLASS_SPLIT'] = each_change[12]

        json.dump(c, f, indent=4)

# iterate through all json file and create a single command training all files
files = os.listdir('./configs')
start_command = "python main.py -tp .\\configs\\"
full_command = ""
for each in files:
    if ".json" in each:
        full_command += start_command+each+";"

print(full_command)
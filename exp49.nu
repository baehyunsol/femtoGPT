cargo run --release -- init --model model-l4-untrained.dat --tokenizer char --tokenizer-data dataset.txt --positional-encoding none --num-tokens 256 --embedding-degree 216 --num-layers 4 --num-heads 8 --case-sensitive;

python3 incremental-training.py 300;

# after training model-l16, I further-trained model-l4 to see how far it can go.
cp model-l4-trained.dat model-l4-further-trained.dat;
cargo run --release -- train --model model-l4-further-trained.dat;
# I have trained 1000 more steps and Ctrl+C. So it's 1300-steps-trained model.

# exp48 and exp49 use the exact same dataset, and the models have the same hyperparams. So we
# can directly compare the losses.
#
# model-l4-trained (last 3 losses)
#   - 2.246
#   - 2.267
#   - 2.209
# model-l4-further-trained
#   - 1.270
#   - 1.236
#   - 1.172
# model-l5-trained
#   - 1.634
#   - 1.773
#   - 1.819
# model-l6-trained
#   - 1.543
#   - 1.471
#   - 1.494
# model-l7-trained
#   - 1.462
#   - 1.515
#   - 1.444
# model-l8-trained
#   - 1.366
#   - 1.487
#   - 1.210
# model-l9-trained
#   - 1.383
#   - 1.219
#   - 1.171
# model-l10-trained
#   - 1.189
#   - 1.274
#   - 1.105
# model-l11-trained
#   - 1.265
#   - 1.049
#   - 1.221
# model-l12-trained
#   - 1.081
#   - 1.152
#   - 1.174
# model-l13-trained
#   - 1.031
#   - 1.132
#   - 1.091
# model-l14-trained
#   - 1.108
#   - 0.989
#   - 1.127
# model-l15-trained
#   - 0.983
#   - 1.112
#   - 1.022
# model-l16-trained
#   - 1.059
#   - 1.063
#   - 1.112
#
# Observations
# Now it's clear that improvements in early iterations are due to more training steps, not inserted layers.
# model-l9-trained is better than model-12-scratch (of exp48). That means incremental training kinda works... right?
# What's most interesting is that model-l4-further-trained is better than every model in exp48. It's even better than
# model-l12-scratch of exp48, which is trained for 2000 steps. Maybe it's because 1) l12 takes tremendously longer to converge
# or 2) my recipe isn't strong enough to train l12.
#
# Comparisons (top-20 most updated layers)
#
# l4-trained vs l4-further-trained
# key: head_1_3_k, cosine: 0.266773
# key: norm_2_bias, cosine: 0.270671
# key: atten_norm_0_bias, cosine: 0.309974
# key: proj_1_bias, cosine: 0.328757
# key: norm_1_bias, cosine: 0.374620
# key: proj_2_bias, cosine: 0.405138
# key: feedforward2_2_bias, cosine: 0.407512
# key: head_3_7_q, cosine: 0.418833
# key: feedforward1_0_bias, cosine: 0.447366
# key: head_3_1_q, cosine: 0.457339
# key: feedforward2_3_bias, cosine: 0.458175
# key: head_2_7_q, cosine: 0.475791
# key: head_1_3_q, cosine: 0.486241
# key: norm_3_bias, cosine: 0.537934
# key: head_1_4_k, cosine: 0.538487
# key: head_2_1_q, cosine: 0.539875
# key: proj_2_weights, cosine: 0.580774
# key: head_2_7_k, cosine: 0.581100
# key: proj_3_weights, cosine: 0.581368
# key: head_2_3_q, cosine: 0.591799
#
# l5-untrained vs l5-trained
# key: head_4_7_k, cosine: 0.382445 (newly inserted layer)
# key: head_4_7_q, cosine: 0.386011 (newly inserted layer)
# key: head_4_6_q, cosine: 0.386892 (newly inserted layer)
# key: head_4_0_q, cosine: 0.389534 (newly inserted layer)
# key: head_4_6_k, cosine: 0.408599 (newly inserted layer)
# key: head_4_0_k, cosine: 0.409180 (newly inserted layer)
# key: head_4_5_q, cosine: 0.412172 (newly inserted layer)
# key: head_4_3_k, cosine: 0.414588 (newly inserted layer)
# key: head_4_3_q, cosine: 0.414895 (newly inserted layer)
# key: head_4_5_k, cosine: 0.424087 (newly inserted layer)
# key: head_4_1_q, cosine: 0.430063 (newly inserted layer)
# key: head_4_2_q, cosine: 0.438149 (newly inserted layer)
# key: head_4_1_k, cosine: 0.448524 (newly inserted layer)
# key: head_4_2_k, cosine: 0.472142 (newly inserted layer)
# key: head_4_4_k, cosine: 0.498376 (newly inserted layer)
# key: head_4_4_q, cosine: 0.538525 (newly inserted layer)
# key: head_2_0_q, cosine: 0.622095
# key: head_2_7_q, cosine: 0.646498
# key: head_2_4_k, cosine: 0.654205
# key: head_1_3_k, cosine: 0.666783
#
# l6-untrained vs l6-trained
# key: head_5_7_q, cosine: 0.540752 (newly inserted layer)
# key: head_5_1_q, cosine: 0.582793 (newly inserted layer)
# key: feedforward2_4_bias, cosine: 0.606505
# key: head_5_7_k, cosine: 0.610196 (newly inserted layer)
# key: head_5_3_q, cosine: 0.627745 (newly inserted layer)
# key: head_5_6_q, cosine: 0.632883 (newly inserted layer)
# key: head_5_2_q, cosine: 0.659714 (newly inserted layer)
# key: norm_2_bias, cosine: 0.669743
# key: head_5_2_k, cosine: 0.682577 (newly inserted layer)
# key: head_5_1_k, cosine: 0.684920 (newly inserted layer)
# key: head_5_6_k, cosine: 0.687148 (newly inserted layer)
# key: head_5_3_k, cosine: 0.701961 (newly inserted layer)
# key: atten_norm_4_bias, cosine: 0.706640
# key: head_5_0_q, cosine: 0.728296 (newly inserted layer)
# key: head_5_0_k, cosine: 0.728317 (newly inserted layer)
# key: head_5_5_q, cosine: 0.736854 (newly inserted layer)
# key: head_5_5_k, cosine: 0.747046 (newly inserted layer)
# key: feedforward1_4_bias, cosine: 0.749201
# key: head_5_4_k, cosine: 0.771741 (newly inserted layer)
# key: head_5_4_q, cosine: 0.781806 (newly inserted layer)
#
# l7-untrained vs l7-trained
# key: atten_norm_5_bias, cosine: 0.761135
# key: feedforward2_5_bias, cosine: 0.797244
# key: head_6_3_k, cosine: 0.862993 (newly inserted layer)
# key: head_6_3_q, cosine: 0.866329 (newly inserted layer)
# key: head_6_6_q, cosine: 0.870628 (newly inserted layer)
# key: proj_5_bias, cosine: 0.872243
# key: feedforward2_4_bias, cosine: 0.879127
# key: feedforward1_6_weights, cosine: 0.881982 (newly inserted layer)
# key: feedforward1_5_bias, cosine: 0.886039
# key: head_6_0_q, cosine: 0.886596 (newly inserted layer)
# key: head_5_3_q, cosine: 0.887070
# key: head_6_6_k, cosine: 0.888984 (newly inserted layer)
# key: head_6_0_k, cosine: 0.895257 (newly inserted layer)
# key: head_6_3_v, cosine: 0.897154 (newly inserted layer)
# key: head_6_6_v, cosine: 0.897867 (newly inserted layer)
# key: head_6_5_v, cosine: 0.898248 (newly inserted layer)
# key: head_5_4_q, cosine: 0.898508
# key: head_6_4_q, cosine: 0.899789 (newly inserted layer)
# key: norm_6_coeff, cosine: 0.900377 (newly inserted layer)
# key: head_6_7_v, cosine: 0.900723 (newly inserted layer)
#
# l8-untrained vs l8-trained
# key: norm_6_bias, cosine: 0.638956
# key: atten_norm_6_bias, cosine: 0.660607
# key: feedforward1_6_bias, cosine: 0.706827
# key: feedforward2_6_bias, cosine: 0.730795
# key: head_6_6_q, cosine: 0.772980
# key: head_6_0_q, cosine: 0.779547
# key: proj_6_bias, cosine: 0.807517
# key: head_6_7_q, cosine: 0.809049
# key: head_6_0_k, cosine: 0.813359
# key: head_6_6_k, cosine: 0.833199
# key: head_6_7_k, cosine: 0.837606
# key: head_6_2_q, cosine: 0.838738
# key: feedforward2_5_bias, cosine: 0.838951
# key: head_6_1_q, cosine: 0.844209
# key: head_6_2_k, cosine: 0.848113
# key: head_6_3_q, cosine: 0.850396
# key: head_6_5_k, cosine: 0.852429
# key: head_6_5_q, cosine: 0.854013
# key: head_6_3_k, cosine: 0.858842
# key: head_6_1_k, cosine: 0.861107
#
# l9-untrained vs l9-trained
# key: atten_norm_7_bias, cosine: 0.771482
# key: feedforward2_7_bias, cosine: 0.811048
# key: norm_7_bias, cosine: 0.811891
# key: feedforward2_6_bias, cosine: 0.823888
# key: feedforward1_6_bias, cosine: 0.832607
# key: feedforward1_7_bias, cosine: 0.837149
# key: atten_norm_6_bias, cosine: 0.854584
# key: proj_7_bias, cosine: 0.860500
# key: head_8_1_k, cosine: 0.870574 (newly inserted layer)
# key: feedforward1_8_weights, cosine: 0.872935 (newly inserted layer)
# key: norm_6_bias, cosine: 0.877356
# key: proj_6_bias, cosine: 0.885546
# key: head_7_6_k, cosine: 0.891353
# key: feedforward2_5_bias, cosine: 0.892500
# key: head_8_3_q, cosine: 0.894734 (newly inserted layer)
# key: head_8_1_q, cosine: 0.897974 (newly inserted layer)
# key: head_8_6_q, cosine: 0.898814 (newly inserted layer)
# key: head_7_6_q, cosine: 0.899545
# key: head_8_6_k, cosine: 0.901250 (newly inserted layer)
# key: head_7_1_q, cosine: 0.901670
# key: head_7_1_k, cosine: 0.902030
#
# l10-untrained vs l10-trained
# key: atten_norm_8_bias, cosine: 0.753796
# key: feedforward2_8_bias, cosine: 0.770872
# key: norm_8_bias, cosine: 0.789576
# key: feedforward1_7_bias, cosine: 0.833195
# key: head_8_6_q, cosine: 0.844034
# key: atten_norm_7_bias, cosine: 0.846471
# key: head_8_6_k, cosine: 0.847864
# key: feedforward1_6_bias, cosine: 0.850517
# key: proj_8_bias, cosine: 0.851360
# key: head_8_7_k, cosine: 0.852733
# key: feedforward2_6_bias, cosine: 0.852840
# key: head_8_4_q, cosine: 0.854638
# key: head_8_4_k, cosine: 0.856879
# key: feedforward1_8_bias, cosine: 0.861135
# key: head_8_7_q, cosine: 0.863297
# key: feedforward2_7_bias, cosine: 0.872102
# key: proj_6_bias, cosine: 0.873514
# key: head_8_3_k, cosine: 0.874146
# key: head_8_3_q, cosine: 0.877013
# key: head_8_1_k, cosine: 0.884398
#
# l11-untrained vs l11-trained
# key: proj_9_bias, cosine: 0.761419
# key: norm_9_bias, cosine: 0.766037
# key: atten_norm_9_bias, cosine: 0.782947
# key: feedforward1_9_bias, cosine: 0.798593
# key: feedforward2_9_bias, cosine: 0.854080
# key: atten_norm_8_bias, cosine: 0.864269
# key: feedforward1_8_bias, cosine: 0.865013
# key: feedforward2_6_bias, cosine: 0.887987
# key: feedforward1_10_weights, cosine: 0.895479 (newly inserted layer)
# key: norm_8_bias, cosine: 0.905632
# key: feedforward2_8_bias, cosine: 0.909293
# key: feedforward1_7_bias, cosine: 0.915549
# key: proj_8_bias, cosine: 0.923544
# key: head_9_7_q, cosine: 0.932092
# key: head_9_7_k, cosine: 0.932753
# key: head_9_4_k, cosine: 0.933454
# key: head_9_4_q, cosine: 0.934480
# key: feedforward2_7_bias, cosine: 0.934842
# key: head_9_0_k, cosine: 0.936723
# key: atten_norm_10_coeff, cosine: 0.937563 (newly inserted layer)
#
# l12-untrained vs l12-trained
# key: atten_norm_10_bias, cosine: 0.623475
# key: feedforward1_10_bias, cosine: 0.735044
# key: proj_10_bias, cosine: 0.743602
# key: norm_10_bias, cosine: 0.752497
# key: feedforward2_10_bias, cosine: 0.761053
# key: feedforward1_8_bias, cosine: 0.844138
# key: norm_9_bias, cosine: 0.845992
# key: feedforward1_9_bias, cosine: 0.846461
# key: atten_norm_8_bias, cosine: 0.861257
# key: atten_norm_9_bias, cosine: 0.881645
# key: proj_9_bias, cosine: 0.881914
# key: feedforward1_11_weights, cosine: 0.899566 (newly inserted layer)
# key: head_10_4_k, cosine: 0.906860
# key: feedforward2_8_bias, cosine: 0.914176
# key: head_10_4_q, cosine: 0.914788
# key: norm_8_bias, cosine: 0.919358
# key: head_11_5_v, cosine: 0.927291 (newly inserted layer)
# key: head_10_2_k, cosine: 0.929694
# key: feedforward1_7_bias, cosine: 0.930239
# key: head_10_2_q, cosine: 0.931385
#
# l13-untrained vs l13-trained
# key: proj_11_bias, cosine: 0.662887
# key: norm_11_bias, cosine: 0.671948
# key: atten_norm_11_bias, cosine: 0.759799
# key: feedforward1_9_bias, cosine: 0.774216
# key: proj_10_bias, cosine: 0.800037
# key: feedforward1_8_bias, cosine: 0.809446
# key: feedforward1_10_bias, cosine: 0.813130
# key: feedforward2_11_bias, cosine: 0.822961
# key: norm_10_bias, cosine: 0.828416
# key: feedforward2_10_bias, cosine: 0.832374
# key: feedforward1_11_bias, cosine: 0.839184
# key: norm_9_bias, cosine: 0.842443
# key: atten_norm_10_bias, cosine: 0.864937
# key: atten_norm_8_bias, cosine: 0.885342
# key: proj_9_bias, cosine: 0.885602
# key: feedforward1_7_bias, cosine: 0.886040
# key: atten_norm_9_bias, cosine: 0.894191
# key: feedforward1_12_weights, cosine: 0.896351 (newly inserted layer)
# key: head_11_6_k, cosine: 0.897763
# key: head_11_6_q, cosine: 0.897778
#
# l14-untrained vs l14-trained
# key: norm_12_bias, cosine: 0.728807
# key: atten_norm_12_bias, cosine: 0.786549
# key: feedforward2_12_bias, cosine: 0.809806
# key: norm_11_bias, cosine: 0.831204
# key: atten_norm_11_bias, cosine: 0.832638
# key: proj_11_bias, cosine: 0.864242
# key: feedforward1_11_bias, cosine: 0.870970
# key: atten_norm_10_bias, cosine: 0.874414
# key: norm_10_bias, cosine: 0.876413
# key: proj_12_bias, cosine: 0.878130
# key: feedforward1_10_bias, cosine: 0.883294
# key: feedforward2_9_bias, cosine: 0.886934
# key: feedforward1_13_weights, cosine: 0.900324 (newly inserted layer)
# key: proj_10_bias, cosine: 0.904295
# key: feedforward2_10_bias, cosine: 0.905857
# key: feedforward2_11_bias, cosine: 0.908467
# key: proj_8_bias, cosine: 0.914699
# key: norm_9_bias, cosine: 0.918450
# key: feedforward1_12_bias, cosine: 0.918554
# key: proj_9_bias, cosine: 0.919495
#
# l15-untrained vs l15-trained
# key: atten_norm_13_bias, cosine: 0.699594
# key: norm_13_bias, cosine: 0.747693
# key: norm_12_bias, cosine: 0.773018
# key: proj_13_bias, cosine: 0.806756
# key: atten_norm_12_bias, cosine: 0.818266
# key: norm_11_bias, cosine: 0.822946
# key: feedforward2_13_bias, cosine: 0.827845
# key: feedforward1_13_bias, cosine: 0.838501
# key: feedforward2_10_bias, cosine: 0.848811
# key: proj_12_bias, cosine: 0.862615
# key: feedforward1_10_bias, cosine: 0.864122
# key: proj_11_bias, cosine: 0.865642
# key: feedforward1_9_bias, cosine: 0.866387
# key: feedforward2_7_bias, cosine: 0.872113
# key: feedforward1_8_bias, cosine: 0.883276
# key: norm_10_bias, cosine: 0.886167
# key: feedforward1_11_bias, cosine: 0.887372
# key: feedforward2_12_bias, cosine: 0.888309
# key: atten_norm_11_bias, cosine: 0.895830
# key: norm_9_bias, cosine: 0.896743
#
# l16-untrained vs l16-trained
# key: norm_14_bias, cosine: 0.641929
# key: atten_norm_14_bias, cosine: 0.705536
# key: proj_14_bias, cosine: 0.771965
# key: atten_norm_13_bias, cosine: 0.785352
# key: norm_13_bias, cosine: 0.800490
# key: feedforward2_13_bias, cosine: 0.808087
# key: feedforward2_14_bias, cosine: 0.811516
# key: feedforward1_14_bias, cosine: 0.827850
# key: norm_12_bias, cosine: 0.850168
# key: proj_11_bias, cosine: 0.855093
# key: feedforward1_13_bias, cosine: 0.875955
# key: feedforward1_12_bias, cosine: 0.877058
# key: proj_13_bias, cosine: 0.881228
# key: feedforward1_11_bias, cosine: 0.886386
# key: atten_norm_12_bias, cosine: 0.891923
# key: proj_12_bias, cosine: 0.903962
# key: feedforward2_12_bias, cosine: 0.906721
# key: feedforward1_15_weights, cosine: 0.911098 (newly inserted layer)
# key: norm_11_bias, cosine: 0.913353
# key: atten_norm_11_bias, cosine: 0.913405

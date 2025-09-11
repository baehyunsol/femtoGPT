cargo run --release -- init --model model-l12-scratch.dat --tokenizer char --tokenizer-data dataset.txt --positional-encoding none --num-tokens 256 --embedding-degree 216 --num-layers 12 --num-heads 8 --case-sensitive;
cargo run --release -- init --model model-l4-untrained.dat --tokenizer char --tokenizer-data dataset.txt --positional-encoding none --num-tokens 256 --embedding-degree 216 --num-layers 4 --num-heads 8 --case-sensitive;

cargo run --release -- train --model model-l12-scratch.dat --steps 2000;

# If you wanna reproduce the result, you have to use c8ab0cc08659 commit of this file: https://github.com/baehyunsol/femtoGPT/blob/c8ab0cc0865940562910575f456040e1c9928ff9/incremental-training.py
python3 incremental-training.py;

# 15 hours on M3 Pro
#
# model-l4-trained (last 3 losses)
#   - 2.855
#   - 2.843
#   - 2.989
#
# model-l5-trained
#   - 2.262
#   - 2.187
#   - 2.177
#
# model-l6-trained
#   - 1.904
#   - 1.841
#   - 1.985
#
# model-l7-trained
#   - 1.780
#   - 1.735
#   - 1.762
#
# model-l8-trained
#   - 1.649
#   - 1.579
#   - 1.774
#
# model-l9-trained
#   - 1.691
#   - 1.615
#   - 1.522
#
# model-l10-trained
#   - 1.817
#   - 1.748
#   - 1.477
#
# model-l11-trained
#   - 1.511
#   - 1.553
#   - 1.533
#
# model-l12-trained
#   - 1.472
#   - 1.567
#   - 1.509
#
# model-l12-scratch
#   - 1.492
#   - 1.686
#   - 1.606
#
# model-l13-trained
#   - 1.525
#   - 1.511
#   - 1.546
#
# Observations
# 1. l5 is better than l4, l6 is better than l5, and l7 is better than l6. Since l7, the model is getting better but the differences are very small.
#    - I can't tell whether the improvement is due to the added layers or training steps (it trains all the layers, not just the new layers).
# 2. The difference between l12-scratch and l12-trained is so small that I cannot tell which one is better.
#    - I compared the inference result of l12-scratch, l12-trained and l13-trained (the dataset is too private so I can't show it to you guys, sorry), and I still couldn't tell the difference.
#
# Comparison between lX-untrained vs lX-trained
# The training scripts adds a layer and trains it for 200 steps in each iteration.
# For example, it adds l5 (which is 6th layer) to l6-untrained and trains it for 200 steps, then it becomes l6-trained.
# I compared the cosine similarities of tensors before and after an iteration.
#
# l5-untrained vs l5-trained
# key: atten_norm_0_bias, cosine: 0.339922
# key: head_1_1_k, cosine: 0.416741
# key: head_1_7_k, cosine: 0.437188
# key: head_1_6_k, cosine: 0.443944
# key: atten_norm_1_bias, cosine: 0.448270
# key: head_1_3_k, cosine: 0.472884
# key: head_1_6_q, cosine: 0.483089
# key: head_3_5_q, cosine: 0.486505
# key: head_1_4_k, cosine: 0.486601
# key: head_1_0_k, cosine: 0.489712
# key: head_1_1_q, cosine: 0.493388
# key: head_1_7_q, cosine: 0.494259
# key: head_1_2_k, cosine: 0.495354
# key: head_1_4_q, cosine: 0.499351
# key: head_1_5_k, cosine: 0.511177
# key: norm_2_bias, cosine: 0.518845
# key: head_4_7_k, cosine: 0.519333 (newly inserted layer)
# key: head_3_3_k, cosine: 0.522948
# key: head_3_5_k, cosine: 0.531120
# key: head_3_7_q, cosine: 0.533969
#
# I gave 200 steps in the first iteration to train the first 4 layers (l0..=l3), and it seems like 200 steps' too short to learn something.
# It's still training the lower layers and the newly inserted layer hasn't been changed much.
#
# l6-untrained vs l6-trained
# key: atten_norm_4_bias, cosine: 0.527030
# key: head_5_1_k, cosine: 0.535203 (newly inserted layer)
# key: head_5_1_q, cosine: 0.545704 (newly inserted layer)
# key: head_5_7_k, cosine: 0.589239 (newly inserted layer)
# key: head_5_5_q, cosine: 0.621975 (newly inserted layer)
# key: head_5_2_k, cosine: 0.646299 (newly inserted layer)
# key: head_5_5_k, cosine: 0.659860 (newly inserted layer)
# key: head_5_7_q, cosine: 0.661696 (newly inserted layer)
# key: head_5_2_q, cosine: 0.665877 (newly inserted layer)
# key: head_5_4_k, cosine: 0.676744 (newly inserted layer)
# key: head_5_4_q, cosine: 0.681858 (newly inserted layer)
# key: head_5_6_k, cosine: 0.688702 (newly inserted layer)
# key: head_5_6_q, cosine: 0.719207 (newly inserted layer)
# key: head_5_3_k, cosine: 0.722880 (newly inserted layer)
# key: head_5_3_q, cosine: 0.726544 (newly inserted layer)
# key: head_5_0_k, cosine: 0.734285 (newly inserted layer)
# key: proj_4_bias, cosine: 0.734984
# key: atten_norm_1_bias, cosine: 0.749584
# key: feedforward2_4_bias, cosine: 0.759025
# key: head_5_0_q, cosine: 0.764231 (newly inserted layer)
#
# Now it's training the newly inserted layer. But their cosine similarities are greater than 0.5, which means the differences are quite small.
#
# l7-untrained vs l7-trained
# key: feedforward2_5_bias, cosine: 0.655559
# key: atten_norm_5_bias, cosine: 0.688522
# key: atten_norm_1_bias, cosine: 0.723305
# key: norm_2_bias, cosine: 0.776589
# key: head_6_1_q, cosine: 0.789198 (newly inserted layer)
# key: proj_5_bias, cosine: 0.796572
# key: head_6_4_k, cosine: 0.801178 (newly inserted layer)
# key: head_6_1_k, cosine: 0.808084 (newly inserted layer)
# key: atten_norm_4_bias, cosine: 0.813981
# key: norm_5_bias, cosine: 0.814215
# key: head_6_7_q, cosine: 0.814467 (newly inserted layer)
# key: head_6_4_q, cosine: 0.819851 (newly inserted layer)
# key: feedforward1_5_bias, cosine: 0.823424
# key: feedforward1_4_bias, cosine: 0.830386
# key: head_6_7_k, cosine: 0.830431 (newly inserted layer)
# key: feedforward2_4_bias, cosine: 0.837123
# key: norm_3_bias, cosine: 0.840962
# key: head_6_2_q, cosine: 0.842370 (newly inserted layer)
# key: head_6_2_k, cosine: 0.852033 (newly inserted layer)
# key: norm_1_bias, cosine: 0.852054
#
# Cosine similarities are even bigger than the previous iteration.
#
# l8-untrained vs l8-trained
# key: norm_6_bias, cosine: 0.720560
# key: proj_6_bias, cosine: 0.768558
# key: atten_norm_6_bias, cosine: 0.793142
# key: feedforward2_6_bias, cosine: 0.818531
# key: atten_norm_5_bias, cosine: 0.824125
# key: head_6_3_q, cosine: 0.844002
# key: head_6_5_q, cosine: 0.850538
# key: head_6_2_k, cosine: 0.852812
# key: head_6_2_q, cosine: 0.854238
# key: feedforward1_6_bias, cosine: 0.854459
# key: head_6_3_k, cosine: 0.860330
# key: head_6_6_q, cosine: 0.864981
# key: head_6_5_k, cosine: 0.866742
# key: head_6_7_q, cosine: 0.868791
# key: head_6_1_q, cosine: 0.871367
# key: head_6_0_q, cosine: 0.872189
# key: feedforward2_5_bias, cosine: 0.875192
# key: head_6_1_k, cosine: 0.877439
# key: proj_5_bias, cosine: 0.881698
# key: feedforward1_5_bias, cosine: 0.882717
#
# Cosine similarities got even bigger. The smallest one is 0.72, that means there's almost no updates in this iteration.
# Also, most updates are in the layer added in the previous iteration.
#
# l9-untrained vs l9-trained
# key: atten_norm_7_bias, cosine: 0.734543
# key: feedforward2_7_bias, cosine: 0.758687
# key: norm_7_bias, cosine: 0.813827
# key: feedforward1_5_bias, cosine: 0.843278
# key: feedforward2_4_bias, cosine: 0.848137
# key: feedforward2_6_bias, cosine: 0.851537
# key: atten_norm_5_bias, cosine: 0.860193
# key: head_7_0_k, cosine: 0.864685
# key: proj_5_bias, cosine: 0.868365
# key: feedforward1_6_bias, cosine: 0.871175
# key: head_7_0_q, cosine: 0.874087
# key: head_7_7_k, cosine: 0.878821
# key: head_7_4_k, cosine: 0.880628
# key: norm_6_bias, cosine: 0.883039
# key: feedforward2_5_bias, cosine: 0.883534
# key: atten_norm_6_bias, cosine: 0.888378
# key: head_7_5_q, cosine: 0.889546
# key: proj_7_bias, cosine: 0.890234
# key: head_7_5_k, cosine: 0.891831
# key: feedforward1_7_bias, cosine: 0.892293
#
# l10-untrained vs l10-trained
# key: atten_norm_8_bias, cosine: 0.703269
# key: norm_8_bias, cosine: 0.750948
# key: feedforward2_8_bias, cosine: 0.767480
# key: feedforward1_8_bias, cosine: 0.825068
# key: norm_7_bias, cosine: 0.840370
# key: feedforward2_6_bias, cosine: 0.843692
# key: proj_7_bias, cosine: 0.870146
# key: atten_norm_7_bias, cosine: 0.872546
# key: feedforward1_5_bias, cosine: 0.874428
# key: feedforward2_7_bias, cosine: 0.874655
# key: proj_8_bias, cosine: 0.875235
# key: proj_6_bias, cosine: 0.883407
# key: head_8_0_k, cosine: 0.887476
# key: head_8_5_k, cosine: 0.898032
# key: head_8_0_q, cosine: 0.903948
# key: head_8_5_q, cosine: 0.904677
# key: feedforward2_4_bias, cosine: 0.907736
# key: atten_norm_5_bias, cosine: 0.911980
# key: head_8_1_q, cosine: 0.912153
# key: norm_6_bias, cosine: 0.919273
#
# l11-untrained vs l11-trained
# key: norm_9_bias, cosine: 0.710778
# key: atten_norm_9_bias, cosine: 0.764016
# key: atten_norm_8_bias, cosine: 0.783286
# key: feedforward2_9_bias, cosine: 0.803341
# key: proj_9_bias, cosine: 0.825153
# key: feedforward2_8_bias, cosine: 0.828455
# key: feedforward1_8_bias, cosine: 0.840112
# key: norm_8_bias, cosine: 0.841891
# key: proj_7_bias, cosine: 0.845783
# key: feedforward1_9_bias, cosine: 0.853702
# key: norm_7_bias, cosine: 0.862000
# key: atten_norm_7_bias, cosine: 0.878863
# key: proj_8_bias, cosine: 0.889974
# key: feedforward2_7_bias, cosine: 0.903312
# key: feedforward2_6_bias, cosine: 0.911630
# key: feedforward1_10_weights, cosine: 0.915244 (newly inserted layer)
# key: feedforward2_4_bias, cosine: 0.921461
# key: feedforward1_7_bias, cosine: 0.930355
# key: feedforward1_6_bias, cosine: 0.932111
# key: feedforward1_4_bias, cosine: 0.941320
#
# l12-untrained vs l12-trained
# key: atten_norm_10_bias, cosine: 0.620468
# key: norm_10_bias, cosine: 0.718939
# key: norm_9_bias, cosine: 0.740225
# key: atten_norm_9_bias, cosine: 0.769636
# key: feedforward2_10_bias, cosine: 0.780370
# key: proj_10_bias, cosine: 0.814236
# key: feedforward1_9_bias, cosine: 0.862938
# key: feedforward1_10_bias, cosine: 0.866415
# key: atten_norm_8_bias, cosine: 0.881286
# key: feedforward1_8_bias, cosine: 0.883210
# key: feedforward2_9_bias, cosine: 0.888870
# key: proj_9_bias, cosine: 0.892611
# key: norm_8_bias, cosine: 0.893513
# key: feedforward2_6_bias, cosine: 0.894172
# key: feedforward2_8_bias, cosine: 0.903717
# key: norm_7_bias, cosine: 0.915177
# key: proj_7_bias, cosine: 0.919833
# key: proj_8_bias, cosine: 0.921401
# key: feedforward1_11_weights, cosine: 0.931528 (newly inserted layer)
# key: atten_norm_7_bias, cosine: 0.938864
#
# l13-untrained vs l13-trained
# key: norm_11_bias, cosine: 0.732317
# key: atten_norm_11_bias, cosine: 0.737439
# key: norm_10_bias, cosine: 0.799210
# key: atten_norm_10_bias, cosine: 0.817751
# key: feedforward1_11_bias, cosine: 0.820890
# key: feedforward2_11_bias, cosine: 0.827698
# key: proj_11_bias, cosine: 0.834501
# key: feedforward2_10_bias, cosine: 0.841548
# key: atten_norm_9_bias, cosine: 0.872551
# key: norm_9_bias, cosine: 0.874334
# key: proj_10_bias, cosine: 0.889468
# key: proj_9_bias, cosine: 0.899700
# key: feedforward2_8_bias, cosine: 0.900419
# key: feedforward2_9_bias, cosine: 0.904560
# key: feedforward1_10_bias, cosine: 0.910606
# key: atten_norm_8_bias, cosine: 0.910994
# key: feedforward1_9_bias, cosine: 0.916762
# key: proj_7_bias, cosine: 0.917596
# key: proj_8_bias, cosine: 0.923616
# key: feedforward1_8_bias, cosine: 0.924001
#
# Since l8, the updates at each iteration are very small. I can understand that it doesn't update the layers added in the previous iterations, but
# it's surprising that it's not updating the newly inserted layers. The new layers are initialized with random numbers, so it has to be drastically
# updated in order for the model to generate valid outputs. I compared the untrained and trained version of models, and verified that the untrained
# ones generate non-sensical outputs, while the trained ones generate sensical outputs. So, it's turning a non-sensical model into a sensical one
# with very small updates: how?

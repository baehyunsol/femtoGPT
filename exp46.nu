cargo run --release -- init --model model-l4.dat --tokenizer char --tokenizer-data dataset.txt --reserve-tokens 32 --positional-encoding none --num-tokens 324 --embedding-degree 288 --num-layers 4 --num-heads 12 --case-sensitive

for _ in 0..8 {
    # initial loss: 6.246
    # step 31 loss: 6.131
    # step 62 loss: 5.373
    # step 93 loss: 3.473
    # step 124 loss: 2.902
    # step 155 loss: 2.689
    # step 186 loss: 2.689
    # NOTE: when I generate tokens with step-186 checkpoint, it only dumps whitespace characters
    # step 217 loss: 2.511
    # step 248 loss: 2.505
    # TIL nu's `0..8` is inclusive so it loops 9 times...
    # step 279 loss: 2.457
    cargo run --release -- train --model model-l4.dat --steps 31;
    sleep 100sec;
}

cargo run --release -- insert-layer --input model-l4.dat --output model-l5-1.dat --insert-at 4;
cargo run --release -- insert-layer --input model-l4.dat --output model-l5-2.dat --insert-at 4;

for _ in 0..8 {
    # initial loss: 3.556
    # step 31 loss: 2.589
    # step 62 loss: 2.400
    # step 93 loss: 2.266
    # step 124 loss: 2.534
    # step 155 loss: 2.336
    # step 186 loss: 2.284
    # step 217 loss: 2.284
    # step 248 loss: 2.193
    # step 279 loss: 2.341
    cargo run --release -- train --model model-l5-1.dat --steps 31;
    sleep 100sec;

    # initial loss: 3.585
    # step 31 loss: 2.774
    # step 62 loss: 2.514
    # step 93 loss: 2.260
    # step 124 loss: 2.432
    # step 155 loss: 2.376
    # step 186 loss: 2.347
    # step 217 loss: 2.484
    # step 248 loss: 2.189
    # step 279 loss: 2.272
    cargo run --release -- train --model model-l5-2.dat --steps 31;
    sleep 100sec;
}

# Choose the better one.
cp model-l5-2.dat model-l5.dat;

cargo run --release -- insert-layer --input model-l5.dat --output model-l6-1.dat --insert-at 5;
cargo run --release -- insert-layer --input model-l5.dat --output model-l6-2.dat --insert-at 5;

for _ in 0..8 {
    cargo run --release -- train --model model-l6-1.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l6-2.dat --steps 31;
    sleep 100sec;
}

# I have enough time, so I'll just train 4 instances and will be back tomorrow.
cargo run --release -- insert-layer --input model-l6-1.dat --output model-l7-1.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-1.dat --output model-l7-2.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-2.dat --output model-l7-3.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-2.dat --output model-l7-4.dat --insert-at 6;

for _ in 0..8 {
    cargo run --release -- train --model model-l7-1.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-2.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-3.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-4.dat --steps 31;
    sleep 100sec;
}

# 2025-09-03: let's see how it went!
#
# model losses (last 3)
# model-l7-1: 2.278, 2.170, 2.235
# model-l7-2: 2.395, 2.201, 2.178
# model-l7-3: 2.284, 2.204, 2.236
# model-l7-4: 2.319, 2.212, 2.222
#
# None of them are superior/inferior than the others.
# It isn't that better than l5 models...
#
# model comparisons (--limit 100)
# NOTE: each layer has 46 tensors
# model-l7-1 vs model-l7-2 (share layer_0..=layer_5)
# key: head_6_8_k, cosine: -0.029687
# key: head_6_7_k, cosine: -0.028440
# key: head_6_2_v, cosine: -0.017046
# key: head_6_1_q, cosine: -0.014651
# key: head_6_9_v, cosine: -0.013313
# key: head_6_9_k, cosine: -0.012381
# key: head_6_1_k, cosine: -0.011583
# key: head_6_6_q, cosine: -0.009429
# key: head_6_5_q, cosine: -0.009060
# key: head_6_5_v, cosine: -0.008156
# key: head_6_2_q, cosine: -0.007967
# key: head_6_8_q, cosine: -0.007718
# key: head_6_3_v, cosine: -0.006958
# key: head_6_6_v, cosine: -0.006772
# key: head_6_10_v, cosine: -0.006656
# key: head_6_10_k, cosine: -0.006104
# key: head_6_4_q, cosine: -0.003691
# key: head_6_5_k, cosine: -0.002118
# key: feedforward1_6_weights, cosine: -0.000033
# key: proj_6_weights, cosine: 0.000252
# key: feedforward2_6_weights, cosine: 0.001906
# key: head_6_2_k, cosine: 0.001934
# key: head_6_7_v, cosine: 0.002128
# key: head_6_0_v, cosine: 0.003103
# key: head_6_11_k, cosine: 0.003146
# key: head_6_11_v, cosine: 0.003608
# key: head_6_3_k, cosine: 0.004246
# key: head_6_3_q, cosine: 0.004548
# key: head_6_0_k, cosine: 0.004842
# key: head_6_9_q, cosine: 0.005062
# key: head_6_4_v, cosine: 0.005345
# key: feedforward1_6_bias, cosine: 0.005461
# key: head_6_4_k, cosine: 0.005590
# key: head_6_10_q, cosine: 0.008966
# key: head_6_0_q, cosine: 0.011535
# key: head_6_8_v, cosine: 0.013634
# key: head_6_1_v, cosine: 0.014036
# key: head_6_7_q, cosine: 0.019480
# key: head_6_11_q, cosine: 0.021994
# key: atten_norm_6_coeff, cosine: 0.031927
# key: head_6_6_k, cosine: 0.046651
# key: proj_6_bias, cosine: 0.047426
# key: norm_6_bias, cosine: 0.112290
# key: norm_6_coeff, cosine: 0.146141
# key: proj_5_bias, cosine: 0.398916 (the first tensor that has nothing to do with the new layer)
# key: atten_norm_6_bias, cosine: 0.481575
# key: atten_norm_5_bias, cosine: 0.592339
# key: feedforward1_5_bias, cosine: 0.664797
# key: feedforward2_5_bias, cosine: 0.697821
# key: norm_5_bias, cosine: 0.758873
# key: feedforward2_6_bias, cosine: 0.770096 (the most similar tensor in the new layer)
# key: proj_4_bias, cosine: 0.784536 (the first tensor that has nothing to do with the new layer and the layer before the new layer)
# key: feedforward2_4_bias, cosine: 0.842041
# key: atten_norm_4_bias, cosine: 0.844138
# key: head_5_7_q, cosine: 0.852577
# key: head_5_8_q, cosine: 0.861617
# key: head_5_10_q, cosine: 0.863469
# key: feedforward1_4_bias, cosine: 0.864998
# key: head_5_11_k, cosine: 0.886475
# key: head_5_1_q, cosine: 0.889464
# key: head_5_10_k, cosine: 0.891087
# key: head_5_2_q, cosine: 0.895753
# key: head_5_0_k, cosine: 0.895923
# key: head_5_11_q, cosine: 0.899986
# key: proj_1_bias, cosine: 0.903262
# key: head_5_0_q, cosine: 0.903736
# key: head_5_7_k, cosine: 0.904732
# key: head_5_8_k, cosine: 0.905502
# key: head_5_3_q, cosine: 0.906153
# key: head_5_1_k, cosine: 0.911788
# key: head_5_5_k, cosine: 0.913885
# key: head_5_2_k, cosine: 0.914774
# key: head_5_5_q, cosine: 0.916232
# key: atten_norm_2_bias, cosine: 0.921178
# key: head_5_6_k, cosine: 0.924661
# key: head_5_6_q, cosine: 0.929014
# key: head_5_3_k, cosine: 0.931738
# key: head_5_4_k, cosine: 0.933977
# key: head_5_9_k, cosine: 0.938807
# key: feedforward2_5_weights, cosine: 0.940529
# key: head_5_4_q, cosine: 0.942644
# key: atten_norm_1_bias, cosine: 0.945134
# key: head_5_9_q, cosine: 0.947548
# key: proj_5_weights, cosine: 0.948300
# key: atten_norm_5_coeff, cosine: 0.949231
# key: head_5_1_v, cosine: 0.951987
# key: norm_5_coeff, cosine: 0.952344
# key: feedforward1_1_weights, cosine: 0.952412
# key: head_5_2_v, cosine: 0.952478
# key: norm_4_bias, cosine: 0.952569
# key: head_5_11_v, cosine: 0.955034
# key: feedforward2_4_weights, cosine: 0.955641
# key: head_5_10_v, cosine: 0.956783
# key: head_5_9_v, cosine: 0.957240
# key: head_5_6_v, cosine: 0.957426
# key: head_5_0_v, cosine: 0.957491
# key: head_4_8_v, cosine: 0.957511
# key: head_5_4_v, cosine: 0.959261
# key: head_5_3_v, cosine: 0.959291
# key: feedforward1_5_weights, cosine: 0.959484
# token_id: 111, token: "├", head: 3, cosine: 0.320543
# token_id: 111, token: "├", head: 10, cosine: 0.450251
# token_id: 111, token: "├", head: 9, cosine: 0.468970
# token_id: 56, token: "┼", head: 8, cosine: 0.489588
# token_id: 111, token: "├", head: 0, cosine: 0.517178
# token_id: 180, token: "✕", head: 8, cosine: 0.527860
# token_id: 438, token: "┌", head: 3, cosine: 0.548378
# token_id: 111, token: "├", head: 4, cosine: 0.549082
# token_id: 111, token: "├", head: 11, cosine: 0.550903
# token_id: 95, token: "ˮ", head: 0, cosine: 0.552416
# token_id: 111, token: "├", head: 8, cosine: 0.552473
# token_id: 111, token: "├", head: 7, cosine: 0.552550
# token_id: 114, token: "┬", head: 4, cosine: 0.568092
# token_id: 180, token: "✕", head: 0, cosine: 0.575827
# token_id: 111, token: "├", head: 1, cosine: 0.592657
# token_id: 367, token: "╴", head: 9, cosine: 0.593080
# token_id: 51, token: "┛", head: 9, cosine: 0.607602
# token_id: 496, token: "▸", head: 3, cosine: 0.608510
# token_id: 119, token: "⊃", head: 11, cosine: 0.611385
# token_id: 51, token: "┛", head: 5, cosine: 0.618081
# token_id: 367, token: "╴", head: 2, cosine: 0.620726
# token_id: 438, token: "┌", head: 11, cosine: 0.621359
# token_id: 496, token: "▸", head: 2, cosine: 0.629080
# token_id: 119, token: "⊃", head: 9, cosine: 0.635894
# token_id: 436, token: "╭", head: 9, cosine: 0.639326
# token_id: 510, token: "、", head: 8, cosine: 0.640341
# token_id: 438, token: "┌", head: 7, cosine: 0.652379
# token_id: 375, token: "”", head: 5, cosine: 0.657479
# token_id: 339, token: "~", head: 5, cosine: 0.658610
# token_id: 294, token: "Z", head: 7, cosine: 0.660302
# token_id: 56, token: "┼", head: 6, cosine: 0.661586
# token_id: 436, token: "╭", head: 1, cosine: 0.664339
# token_id: 496, token: "▸", head: 5, cosine: 0.677990
# token_id: 436, token: "╭", head: 10, cosine: 0.680212
# token_id: 111, token: "├", head: 5, cosine: 0.681063
# token_id: 436, token: "╭", head: 3, cosine: 0.684216
# token_id: 56, token: "┼", head: 10, cosine: 0.684541
# token_id: 119, token: "⊃", head: 5, cosine: 0.685506
# token_id: 438, token: "┌", head: 4, cosine: 0.685565
# token_id: 436, token: "╭", head: 11, cosine: 0.689603
# token_id: 119, token: "⊃", head: 7, cosine: 0.694218
# token_id: 180, token: "✕", head: 4, cosine: 0.694649
# token_id: 168, token: "→", head: 2, cosine: 0.694893
# token_id: 119, token: "⊃", head: 8, cosine: 0.695728
# token_id: 438, token: "┌", head: 9, cosine: 0.695959
# token_id: 496, token: "▸", head: 9, cosine: 0.696805
# token_id: 56, token: "┼", head: 3, cosine: 0.698836
# token_id: 510, token: "、", head: 0, cosine: 0.701593
# token_id: 161, token: "━", head: 4, cosine: 0.703804
# token_id: 51, token: "┛", head: 10, cosine: 0.704717
# token_id: 51, token: "┛", head: 11, cosine: 0.705175
# token_id: 56, token: "┼", head: 1, cosine: 0.708135
# token_id: 39, token: "⟋", head: 8, cosine: 0.710703
# token_id: 56, token: "┼", head: 4, cosine: 0.712794
# token_id: 496, token: "▸", head: 7, cosine: 0.712816
# token_id: 119, token: "⊃", head: 3, cosine: 0.715339
# token_id: 180, token: "✕", head: 10, cosine: 0.716773
# token_id: 406, token: "⁄", head: 6, cosine: 0.718257
# token_id: 63, token: "Y", head: 6, cosine: 0.719905
# token_id: 39, token: "⟋", head: 10, cosine: 0.722846
# token_id: 11, token: "く", head: 0, cosine: 0.726016
# token_id: 170, token: "\\", head: 10, cosine: 0.732770
# token_id: 33, token: "和", head: 4, cosine: 0.734713
# token_id: 114, token: "┬", head: 3, cosine: 0.734782
# token_id: 436, token: "╭", head: 6, cosine: 0.735577
# token_id: 202, token: "㇓", head: 9, cosine: 0.739464
# token_id: 294, token: "Z", head: 6, cosine: 0.739513
# token_id: 111, token: "├", head: 2, cosine: 0.739706
# token_id: 63, token: "Y", head: 3, cosine: 0.741209
# token_id: 294, token: "Z", head: 3, cosine: 0.743835
# token_id: 234, token: "＆", head: 6, cosine: 0.744612
# token_id: 438, token: "┌", head: 1, cosine: 0.747807
# token_id: 234, token: "＆", head: 1, cosine: 0.749203
# token_id: 17, token: "﬩", head: 6, cosine: 0.749688
# token_id: 111, token: "├", head: 6, cosine: 0.749690
# token_id: 119, token: "⊃", head: 1, cosine: 0.751236
# token_id: 152, token: "│", head: 5, cosine: 0.752656
# token_id: 180, token: "✕", head: 1, cosine: 0.752688
# token_id: 11, token: "く", head: 9, cosine: 0.753586
# token_id: 236, token: "ײ", head: 3, cosine: 0.754606
# token_id: 246, token: "᛭", head: 2, cosine: 0.755899
# token_id: 39, token: "⟋", head: 0, cosine: 0.757285
# token_id: 119, token: "⊃", head: 4, cosine: 0.757827
# token_id: 168, token: "→", head: 11, cosine: 0.758589
# token_id: 17, token: "﬩", head: 2, cosine: 0.758695
# token_id: 11, token: "く", head: 1, cosine: 0.758807
# token_id: 294, token: "Z", head: 9, cosine: 0.759445
# token_id: 266, token: "╰", head: 1, cosine: 0.761968
# token_id: 209, token: "┃", head: 10, cosine: 0.762392
# token_id: 438, token: "┌", head: 5, cosine: 0.762435
# token_id: 234, token: "＆", head: 4, cosine: 0.763843
# token_id: 508, token: "»", head: 10, cosine: 0.764230
# token_id: 234, token: "＆", head: 7, cosine: 0.765354
# token_id: 393, token: "❬", head: 10, cosine: 0.765469
# token_id: 234, token: "＆", head: 0, cosine: 0.765766
# token_id: 510, token: "、", head: 7, cosine: 0.767237
# token_id: 400, token: "└", head: 11, cosine: 0.767525
# token_id: 266, token: "╰", head: 3, cosine: 0.767923
# token_id: 202, token: "㇓", head: 10, cosine: 0.768330
# token_id: 406, token: "⁄", head: 10, cosine: 0.768935
#
# model-l7-1 vs model-l7-3 (share layer_0..=layer_4)
# key: head_5_3_q, cosine: -0.044442
# key: head_5_10_q, cosine: -0.037973
# key: norm_6_coeff, cosine: -0.028030
# key: head_5_10_k, cosine: -0.025417
# key: head_5_3_k, cosine: -0.024340
# key: head_5_4_q, cosine: -0.023538
# key: head_6_9_k, cosine: -0.021564
# key: head_6_5_v, cosine: -0.020276
# key: head_5_6_k, cosine: -0.019839
# key: head_6_4_v, cosine: -0.017500
# key: head_6_6_q, cosine: -0.016469
# key: head_5_4_k, cosine: -0.015593
# key: head_5_5_v, cosine: -0.015344
# key: head_6_9_q, cosine: -0.013939
# key: head_5_9_v, cosine: -0.013205
# key: feedforward1_5_bias, cosine: -0.012797
# key: head_5_1_k, cosine: -0.012326
# key: head_5_6_q, cosine: -0.010607
# key: head_6_0_v, cosine: -0.010557
# key: head_5_1_v, cosine: -0.010124
# key: head_6_7_q, cosine: -0.009489
# key: head_6_1_v, cosine: -0.008311
# key: head_5_1_q, cosine: -0.008300
# key: head_5_10_v, cosine: -0.008285
# key: head_6_5_q, cosine: -0.008255
# key: head_6_1_q, cosine: -0.008116
# key: head_6_7_k, cosine: -0.008032
# key: head_6_10_q, cosine: -0.006939
# key: head_5_5_k, cosine: -0.006855
# key: head_5_0_v, cosine: -0.006653
# key: head_5_11_q, cosine: -0.006193
# key: head_6_3_k, cosine: -0.005821
# key: head_6_0_k, cosine: -0.005116
# key: head_5_2_v, cosine: -0.004831
# key: head_6_7_v, cosine: -0.003873
# key: head_6_11_v, cosine: -0.003210
# key: head_6_2_k, cosine: -0.003071
# key: head_6_9_v, cosine: -0.002382
# key: head_6_11_q, cosine: -0.002111
# key: head_5_5_q, cosine: -0.001833
# key: feedforward1_6_weights, cosine: -0.000521
# key: head_6_8_v, cosine: -0.000224
# key: head_6_0_q, cosine: 0.000126
# key: feedforward2_6_weights, cosine: 0.000184
# key: head_5_6_v, cosine: 0.000578
# key: feedforward1_5_weights, cosine: 0.000786
# key: proj_6_weights, cosine: 0.000882
# key: head_6_2_v, cosine: 0.001172
# key: feedforward2_5_weights, cosine: 0.001202
# key: head_6_4_k, cosine: 0.001992
# key: head_6_10_k, cosine: 0.002092
# key: head_6_4_q, cosine: 0.002217
# key: head_6_8_k, cosine: 0.002317
# key: head_5_8_k, cosine: 0.003067
# key: head_6_1_k, cosine: 0.004303
# key: head_6_3_q, cosine: 0.004833
# key: head_6_3_v, cosine: 0.006783
# key: head_5_0_k, cosine: 0.006830
# key: head_6_2_q, cosine: 0.006883
# key: head_5_7_v, cosine: 0.007058
# key: head_5_2_k, cosine: 0.007058
# key: proj_5_weights, cosine: 0.007652
# key: feedforward1_6_bias, cosine: 0.008331
# key: head_5_3_v, cosine: 0.008963
# key: head_6_11_k, cosine: 0.009252
# key: head_5_11_k, cosine: 0.010903
# key: head_5_4_v, cosine: 0.010941
# key: head_5_11_v, cosine: 0.011845
# key: head_6_6_k, cosine: 0.011936
# key: head_5_8_q, cosine: 0.012317
# key: head_5_7_k, cosine: 0.012667
# key: head_5_2_q, cosine: 0.014361
# key: head_5_0_q, cosine: 0.014923
# key: head_5_7_q, cosine: 0.015250
# key: head_6_8_q, cosine: 0.015379
# key: head_6_6_v, cosine: 0.018912
# key: head_6_5_k, cosine: 0.023646
# key: head_5_9_q, cosine: 0.025297
# key: head_6_10_v, cosine: 0.025373
# key: head_5_9_k, cosine: 0.026904
# key: head_5_8_v, cosine: 0.027596
# key: atten_norm_6_coeff, cosine: 0.050639
# key: proj_6_bias, cosine: 0.060868
# key: norm_6_bias, cosine: 0.067191
# key: norm_5_coeff, cosine: 0.120395
# key: atten_norm_5_coeff, cosine: 0.135339
# key: proj_5_bias, cosine: 0.180354
# key: norm_5_bias, cosine: 0.232171
# key: atten_norm_5_bias, cosine: 0.382961
# key: feedforward2_4_bias, cosine: 0.481051 (the first tensor that has nothing to do with the new layers)
# key: atten_norm_6_bias, cosine: 0.483065
# key: feedforward1_4_bias, cosine: 0.598038
# key: proj_4_bias, cosine: 0.659206
# key: proj_1_bias, cosine: 0.676755 (the first tensor that has nothing to do with the new layers and the layer before the new layers)
# key: feedforward2_5_bias, cosine: 0.678419
# key: atten_norm_4_bias, cosine: 0.681726
# key: feedforward2_6_bias, cosine: 0.714039 (the most similar tensor in the new layers)
# key: head_4_8_q, cosine: 0.773732
# key: proj_1_weights, cosine: 0.863856
# key: feedforward2_3_bias, cosine: 0.872443
# token_id: 180, token: "✕", head: 0, cosine: 0.126857
# token_id: 426, token: "𐊛", head: 9, cosine: 0.168085
# token_id: 17, token: "﬩", head: 6, cosine: 0.182252
# token_id: 416, token: "►", head: 10, cosine: 0.250198
# token_id: 17, token: "﬩", head: 2, cosine: 0.251000
# token_id: 111, token: "├", head: 3, cosine: 0.282039
# token_id: 416, token: "►", head: 4, cosine: 0.298854
# token_id: 114, token: "┬", head: 4, cosine: 0.349075
# token_id: 180, token: "✕", head: 4, cosine: 0.368377
# token_id: 119, token: "⊃", head: 11, cosine: 0.379757
# token_id: 438, token: "┌", head: 4, cosine: 0.385865
# token_id: 180, token: "✕", head: 10, cosine: 0.396410
# token_id: 180, token: "✕", head: 9, cosine: 0.397001
# token_id: 119, token: "⊃", head: 5, cosine: 0.411966
# token_id: 17, token: "﬩", head: 3, cosine: 0.419647
# token_id: 341, token: "’", head: 7, cosine: 0.432325
# token_id: 51, token: "┛", head: 10, cosine: 0.436983
# token_id: 416, token: "►", head: 1, cosine: 0.439858
# token_id: 416, token: "►", head: 7, cosine: 0.448754
# token_id: 416, token: "►", head: 0, cosine: 0.458469
# token_id: 51, token: "┛", head: 5, cosine: 0.460171
# token_id: 510, token: "、", head: 8, cosine: 0.471475
# token_id: 119, token: "⊃", head: 4, cosine: 0.478235
# token_id: 350, token: "˴", head: 0, cosine: 0.492794
# token_id: 168, token: "→", head: 4, cosine: 0.496264
# token_id: 382, token: "—", head: 5, cosine: 0.499897
# token_id: 436, token: "╭", head: 3, cosine: 0.501318
# token_id: 147, token: "\u{a0}", head: 2, cosine: 0.515223
# token_id: 114, token: "┬", head: 7, cosine: 0.519028
# token_id: 119, token: "⊃", head: 1, cosine: 0.521669
# token_id: 119, token: "⊃", head: 3, cosine: 0.523647
# token_id: 147, token: "\u{a0}", head: 4, cosine: 0.526710
# token_id: 119, token: "⊃", head: 8, cosine: 0.530667
# token_id: 129, token: "﴿", head: 7, cosine: 0.536396
# token_id: 111, token: "├", head: 0, cosine: 0.536457
# token_id: 180, token: "✕", head: 5, cosine: 0.537450
# token_id: 394, token: "┐", head: 11, cosine: 0.538200
# token_id: 17, token: "﬩", head: 8, cosine: 0.539992
# token_id: 128, token: "ߺ", head: 8, cosine: 0.542053
# token_id: 168, token: "→", head: 0, cosine: 0.545115
# token_id: 27, token: "·", head: 4, cosine: 0.545127
# token_id: 27, token: "·", head: 9, cosine: 0.546635
# token_id: 95, token: "ˮ", head: 0, cosine: 0.552416
# token_id: 17, token: "﬩", head: 9, cosine: 0.554220
# token_id: 199, token: "┤", head: 1, cosine: 0.554578
# token_id: 17, token: "﬩", head: 7, cosine: 0.555929
# token_id: 339, token: "~", head: 10, cosine: 0.556254
# token_id: 111, token: "├", head: 1, cosine: 0.556421
# token_id: 341, token: "’", head: 6, cosine: 0.560221
# token_id: 180, token: "✕", head: 1, cosine: 0.561614
# token_id: 338, token: "┗", head: 7, cosine: 0.563416
# token_id: 17, token: "﬩", head: 0, cosine: 0.563548
# token_id: 483, token: "░", head: 4, cosine: 0.563803
# token_id: 119, token: "⊃", head: 6, cosine: 0.564272
# token_id: 119, token: "⊃", head: 2, cosine: 0.567715
# token_id: 68, token: "@", head: 9, cosine: 0.572596
# token_id: 114, token: "┬", head: 8, cosine: 0.573796
# token_id: 426, token: "𐊛", head: 7, cosine: 0.573886
# token_id: 416, token: "►", head: 6, cosine: 0.580162
# token_id: 168, token: "→", head: 1, cosine: 0.581388
# token_id: 350, token: "˴", head: 8, cosine: 0.581571
# token_id: 208, token: "‹", head: 10, cosine: 0.586691
# token_id: 341, token: "’", head: 1, cosine: 0.587734
# token_id: 341, token: "’", head: 3, cosine: 0.595468
# token_id: 119, token: "⊃", head: 9, cosine: 0.597178
# token_id: 51, token: "┛", head: 11, cosine: 0.598812
# token_id: 17, token: "﬩", head: 4, cosine: 0.599313
# token_id: 111, token: "├", head: 8, cosine: 0.601903
# token_id: 341, token: "’", head: 10, cosine: 0.602421
# token_id: 416, token: "►", head: 5, cosine: 0.604358
# token_id: 17, token: "﬩", head: 1, cosine: 0.607945
# token_id: 180, token: "✕", head: 8, cosine: 0.610542
# token_id: 339, token: "~", head: 8, cosine: 0.611050
# token_id: 496, token: "▸", head: 3, cosine: 0.616141
# token_id: 147, token: "\u{a0}", head: 9, cosine: 0.617060
# token_id: 111, token: "├", head: 9, cosine: 0.618403
# token_id: 152, token: "│", head: 5, cosine: 0.619551
# token_id: 421, token: "²", head: 9, cosine: 0.619956
# token_id: 350, token: "˴", head: 7, cosine: 0.624854
# token_id: 472, token: "\u{3000}", head: 1, cosine: 0.625906
# token_id: 68, token: "@", head: 5, cosine: 0.630164
# token_id: 436, token: "╭", head: 6, cosine: 0.630402
# token_id: 350, token: "˴", head: 4, cosine: 0.632130
# token_id: 114, token: "┬", head: 0, cosine: 0.634296
# token_id: 314, token: "－", head: 5, cosine: 0.634678
# token_id: 51, token: "┛", head: 3, cosine: 0.639919
# token_id: 260, token: "＋", head: 6, cosine: 0.640997
# token_id: 270, token: "︖", head: 2, cosine: 0.641644
# token_id: 147, token: "\u{a0}", head: 5, cosine: 0.641736
# token_id: 180, token: "✕", head: 2, cosine: 0.641770
# token_id: 339, token: "~", head: 3, cosine: 0.642214
# token_id: 114, token: "┬", head: 10, cosine: 0.643224
# token_id: 180, token: "✕", head: 7, cosine: 0.643828
# token_id: 294, token: "Z", head: 7, cosine: 0.646295
# token_id: 180, token: "✕", head: 3, cosine: 0.648542
# token_id: 469, token: "［", head: 10, cosine: 0.650319
# token_id: 314, token: "－", head: 7, cosine: 0.651954
# token_id: 416, token: "►", head: 3, cosine: 0.653126
# token_id: 93, token: "\u{2009}", head: 3, cosine: 0.654478
# token_id: 111, token: "├", head: 10, cosine: 0.654764
#
# model-l7-1 vs model-l7-4 (share layer_0..=layer_4)
# key: norm_6_coeff, cosine: -0.057999
# key: norm_6_bias, cosine: -0.046285
# key: head_5_3_q, cosine: -0.044390
# key: head_5_10_q, cosine: -0.038121
# key: head_6_11_k, cosine: -0.034772
# key: head_6_6_k, cosine: -0.027579
# key: head_5_4_q, cosine: -0.023305
# key: head_5_3_k, cosine: -0.023154
# key: head_5_10_k, cosine: -0.020723
# key: head_6_7_k, cosine: -0.020540
# key: head_6_7_q, cosine: -0.017760
# key: head_5_6_k, cosine: -0.016221
# key: head_5_1_v, cosine: -0.015274
# key: head_6_1_q, cosine: -0.014344
# key: head_5_4_k, cosine: -0.014161
# key: head_5_5_v, cosine: -0.012480
# key: head_5_9_v, cosine: -0.011091
# key: head_6_8_v, cosine: -0.010796
# key: head_6_2_v, cosine: -0.010210
# key: head_6_11_q, cosine: -0.009653
# key: head_5_1_k, cosine: -0.008602
# key: head_6_4_k, cosine: -0.008498
# key: head_5_6_q, cosine: -0.008121
# key: head_6_2_k, cosine: -0.007684
# key: head_5_10_v, cosine: -0.007673
# key: head_5_11_q, cosine: -0.007652
# key: head_5_0_v, cosine: -0.006801
# key: head_5_5_k, cosine: -0.006708
# key: head_6_10_v, cosine: -0.005377
# key: head_6_11_v, cosine: -0.004665
# key: head_5_1_q, cosine: -0.004143
# key: head_6_1_v, cosine: -0.002390
# key: feedforward2_6_weights, cosine: -0.002363
# key: head_6_9_q, cosine: -0.001906
# key: head_5_2_v, cosine: -0.001007
# key: head_6_10_k, cosine: -0.000900
# key: head_6_5_k, cosine: -0.000862
# key: head_6_4_v, cosine: -0.000544
# key: head_5_2_k, cosine: -0.000381
# key: head_6_4_q, cosine: 0.000084
# key: feedforward1_5_weights, cosine: 0.000709
# key: head_5_6_v, cosine: 0.000969
# key: feedforward2_5_weights, cosine: 0.001376
# key: feedforward1_6_weights, cosine: 0.001536
# key: head_5_8_k, cosine: 0.001832
# key: head_5_5_q, cosine: 0.001865
# key: head_6_8_k, cosine: 0.002143
# key: head_5_11_k, cosine: 0.003031
# key: head_6_0_v, cosine: 0.003062
# key: head_6_9_v, cosine: 0.003824
# key: feedforward1_5_bias, cosine: 0.003994
# key: proj_6_weights, cosine: 0.004047
# key: head_6_8_q, cosine: 0.004591
# key: head_6_5_q, cosine: 0.004633
# key: head_6_9_k, cosine: 0.004654
# key: head_5_7_k, cosine: 0.004780
# key: head_6_7_v, cosine: 0.005298
# key: head_5_0_k, cosine: 0.005444
# key: head_6_0_q, cosine: 0.005521
# key: head_5_8_q, cosine: 0.005541
# key: feedforward1_6_bias, cosine: 0.005881
# key: head_6_3_q, cosine: 0.005954
# key: head_6_6_q, cosine: 0.006197
# key: head_6_2_q, cosine: 0.007058
# key: head_5_3_v, cosine: 0.007605
# key: head_6_1_k, cosine: 0.007849
# key: head_5_7_v, cosine: 0.008255
# key: head_6_10_q, cosine: 0.008389
# key: head_5_4_v, cosine: 0.009129
# key: head_5_2_q, cosine: 0.009227
# key: proj_5_weights, cosine: 0.009371
# key: head_5_11_v, cosine: 0.011825
# key: head_5_7_q, cosine: 0.011914
# key: head_6_3_k, cosine: 0.015355
# key: head_5_0_q, cosine: 0.015445
# key: head_5_9_k, cosine: 0.018486
# key: head_6_3_v, cosine: 0.019399
# key: head_5_8_v, cosine: 0.019924
# key: head_6_0_k, cosine: 0.019998
# key: head_6_5_v, cosine: 0.020591
# key: head_6_6_v, cosine: 0.027096
# key: head_5_9_q, cosine: 0.028322
# key: proj_6_bias, cosine: 0.056733
# key: atten_norm_6_coeff, cosine: 0.094110
# key: norm_5_coeff, cosine: 0.131943
# key: proj_5_bias, cosine: 0.142528
# key: atten_norm_5_coeff, cosine: 0.146665
# key: norm_5_bias, cosine: 0.192628
# key: atten_norm_6_bias, cosine: 0.239228
# key: atten_norm_5_bias, cosine: 0.424696
# key: feedforward2_4_bias, cosine: 0.491174 (the first tensor that has nothing to do with the new layers)
# key: feedforward1_4_bias, cosine: 0.618275
# key: feedforward2_6_bias, cosine: 0.622819
# key: atten_norm_4_bias, cosine: 0.661437
# key: proj_4_bias, cosine: 0.664989
# key: feedforward2_5_bias, cosine: 0.690687 (the most similar tensor in the new layers)
# key: proj_1_bias, cosine: 0.693621 (the first tensor that has nothing to do with the new layers and the layer before the new layers)
# key: head_4_8_q, cosine: 0.758854
# key: proj_1_weights, cosine: 0.864441
# key: feedforward2_3_bias, cosine: 0.875199
# token_id: 426, token: "𐊛", head: 9, cosine: 0.168085
# token_id: 17, token: "﬩", head: 6, cosine: 0.182252
# token_id: 180, token: "✕", head: 0, cosine: 0.219065
# token_id: 367, token: "╴", head: 9, cosine: 0.224631
# token_id: 17, token: "﬩", head: 2, cosine: 0.251000
# token_id: 341, token: "’", head: 7, cosine: 0.251443
# token_id: 180, token: "✕", head: 8, cosine: 0.299962
# token_id: 496, token: "▸", head: 3, cosine: 0.369258
# token_id: 119, token: "⊃", head: 11, cosine: 0.379757
# token_id: 510, token: "、", head: 8, cosine: 0.393179
# token_id: 111, token: "├", head: 3, cosine: 0.411012
# token_id: 119, token: "⊃", head: 5, cosine: 0.411966
# token_id: 496, token: "▸", head: 10, cosine: 0.418063
# token_id: 17, token: "﬩", head: 3, cosine: 0.419647
# token_id: 180, token: "✕", head: 7, cosine: 0.431524
# token_id: 266, token: "╰", head: 3, cosine: 0.441623
# token_id: 180, token: "✕", head: 10, cosine: 0.463633
# token_id: 180, token: "✕", head: 4, cosine: 0.464825
# token_id: 180, token: "✕", head: 1, cosine: 0.466095
# token_id: 180, token: "✕", head: 9, cosine: 0.467875
# token_id: 119, token: "⊃", head: 4, cosine: 0.478235
# token_id: 51, token: "┛", head: 10, cosine: 0.484898
# token_id: 496, token: "▸", head: 5, cosine: 0.490484
# token_id: 350, token: "˴", head: 0, cosine: 0.492794
# token_id: 148, token: "╾", head: 6, cosine: 0.495577
# token_id: 119, token: "⊃", head: 1, cosine: 0.521669
# token_id: 496, token: "▸", head: 9, cosine: 0.523058
# token_id: 51, token: "┛", head: 5, cosine: 0.523624
# token_id: 119, token: "⊃", head: 3, cosine: 0.523647
# token_id: 119, token: "⊃", head: 8, cosine: 0.530667
# token_id: 129, token: "﴿", head: 7, cosine: 0.536396
# token_id: 111, token: "├", head: 0, cosine: 0.538894
# token_id: 17, token: "﬩", head: 8, cosine: 0.539992
# token_id: 148, token: "╾", head: 0, cosine: 0.540342
# token_id: 128, token: "ߺ", head: 8, cosine: 0.542053
# token_id: 438, token: "┌", head: 4, cosine: 0.543956
# token_id: 147, token: "\u{a0}", head: 4, cosine: 0.548765
# token_id: 367, token: "╴", head: 2, cosine: 0.550752
# token_id: 95, token: "ˮ", head: 0, cosine: 0.552416
# token_id: 17, token: "﬩", head: 9, cosine: 0.554220
# token_id: 17, token: "﬩", head: 7, cosine: 0.555929
# token_id: 382, token: "—", head: 5, cosine: 0.556089
# token_id: 339, token: "~", head: 8, cosine: 0.559505
# token_id: 17, token: "﬩", head: 0, cosine: 0.563548
# token_id: 483, token: "░", head: 4, cosine: 0.563803
# token_id: 119, token: "⊃", head: 6, cosine: 0.564272
# token_id: 147, token: "\u{a0}", head: 11, cosine: 0.565839
# token_id: 119, token: "⊃", head: 2, cosine: 0.567715
# token_id: 339, token: "~", head: 5, cosine: 0.570878
# token_id: 180, token: "✕", head: 5, cosine: 0.571595
# token_id: 426, token: "𐊛", head: 7, cosine: 0.573886
# token_id: 148, token: "╾", head: 2, cosine: 0.576686
# token_id: 112, token: "⚬", head: 11, cosine: 0.579366
# token_id: 350, token: "˴", head: 8, cosine: 0.581571
# token_id: 208, token: "‹", head: 10, cosine: 0.586691
# token_id: 148, token: "╾", head: 10, cosine: 0.587768
# token_id: 510, token: "、", head: 7, cosine: 0.588237
# token_id: 177, token: "❨", head: 2, cosine: 0.592087
# token_id: 111, token: "├", head: 1, cosine: 0.592194
# token_id: 177, token: "❨", head: 7, cosine: 0.596637
# token_id: 496, token: "▸", head: 8, cosine: 0.596690
# token_id: 119, token: "⊃", head: 9, cosine: 0.597178
# token_id: 17, token: "﬩", head: 4, cosine: 0.599313
# token_id: 177, token: "❨", head: 8, cosine: 0.600117
# token_id: 96, token: "ᚲ", head: 11, cosine: 0.601105
# token_id: 168, token: "→", head: 0, cosine: 0.602088
# token_id: 180, token: "✕", head: 2, cosine: 0.603154
# token_id: 339, token: "~", head: 10, cosine: 0.604273
# token_id: 112, token: "⚬", head: 10, cosine: 0.605599
# token_id: 114, token: "┬", head: 4, cosine: 0.607142
# token_id: 17, token: "﬩", head: 1, cosine: 0.607945
# token_id: 394, token: "┐", head: 11, cosine: 0.609496
# token_id: 436, token: "╭", head: 6, cosine: 0.610737
# token_id: 51, token: "┛", head: 11, cosine: 0.613611
# token_id: 168, token: "→", head: 2, cosine: 0.616443
# token_id: 68, token: "@", head: 9, cosine: 0.616843
# token_id: 168, token: "→", head: 4, cosine: 0.617010
# token_id: 436, token: "╭", head: 3, cosine: 0.617847
# token_id: 510, token: "、", head: 0, cosine: 0.620238
# token_id: 148, token: "╾", head: 3, cosine: 0.623287
# token_id: 341, token: "’", head: 3, cosine: 0.624428
# token_id: 350, token: "˴", head: 7, cosine: 0.624854
# token_id: 472, token: "\u{3000}", head: 1, cosine: 0.625906
# token_id: 68, token: "@", head: 3, cosine: 0.627980
# token_id: 339, token: "~", head: 4, cosine: 0.627992
# token_id: 147, token: "\u{a0}", head: 5, cosine: 0.627996
# token_id: 367, token: "╴", head: 10, cosine: 0.628840
# token_id: 147, token: "\u{a0}", head: 2, cosine: 0.629742
# token_id: 350, token: "˴", head: 4, cosine: 0.632130
# token_id: 112, token: "⚬", head: 3, cosine: 0.633469
# token_id: 266, token: "╰", head: 8, cosine: 0.638669
# token_id: 112, token: "⚬", head: 7, cosine: 0.640590
# token_id: 438, token: "┌", head: 3, cosine: 0.640738
# token_id: 260, token: "＋", head: 6, cosine: 0.640997
# token_id: 270, token: "︖", head: 2, cosine: 0.641644
# token_id: 148, token: "╾", head: 4, cosine: 0.643647
# token_id: 180, token: "✕", head: 11, cosine: 0.643681
# token_id: 253, token: "^", head: 7, cosine: 0.644220
# token_id: 416, token: "►", head: 4, cosine: 0.645201
# token_id: 168, token: "→", head: 1, cosine: 0.646313
#
# model-l7-2 vs model-l7-3 (share layer_0..=layer_4)
# key: head_5_3_q, cosine: -0.038278
# key: head_6_9_q, cosine: -0.025957
# key: head_5_3_k, cosine: -0.025927
# key: norm_6_coeff, cosine: -0.023602
# key: head_5_10_q, cosine: -0.022973
# key: head_5_4_q, cosine: -0.022553
# key: head_5_10_k, cosine: -0.022229
# key: head_6_10_v, cosine: -0.021692
# key: head_5_4_k, cosine: -0.016789
# key: head_5_6_k, cosine: -0.016387
# key: head_5_1_v, cosine: -0.016070
# key: head_6_5_v, cosine: -0.015213
# key: head_6_3_q, cosine: -0.014750
# key: head_5_1_k, cosine: -0.013893
# key: head_5_5_v, cosine: -0.013624
# key: head_6_7_k, cosine: -0.012657
# key: head_5_2_v, cosine: -0.011595
# key: head_6_7_v, cosine: -0.009896
# key: head_6_10_k, cosine: -0.009885
# key: head_5_0_v, cosine: -0.009785
# key: head_6_4_v, cosine: -0.009712
# key: head_5_6_q, cosine: -0.008873
# key: head_6_1_v, cosine: -0.007544
# key: head_5_10_v, cosine: -0.006227
# key: head_6_2_k, cosine: -0.006148
# key: head_6_4_k, cosine: -0.005998
# key: head_5_5_k, cosine: -0.005678
# key: head_6_9_v, cosine: -0.005560
# key: head_6_11_q, cosine: -0.005246
# key: head_5_1_q, cosine: -0.004498
# key: head_5_9_v, cosine: -0.004440
# key: feedforward1_6_weights, cosine: -0.003100
# key: head_6_5_q, cosine: -0.002822
# key: head_6_1_k, cosine: -0.002606
# key: feedforward2_6_weights, cosine: -0.000928
# key: proj_6_weights, cosine: -0.000250
# key: head_6_9_k, cosine: -0.000042
# key: head_6_11_v, cosine: 0.000639
# key: feedforward2_5_weights, cosine: 0.000845
# key: head_5_5_q, cosine: 0.000923
# key: feedforward1_5_weights, cosine: 0.001176
# key: head_6_0_v, cosine: 0.002795
# key: head_5_11_q, cosine: 0.002932
# key: head_5_3_v, cosine: 0.002976
# key: head_5_6_v, cosine: 0.003267
# key: head_6_0_k, cosine: 0.003277
# key: head_6_8_k, cosine: 0.003775
# key: head_6_1_q, cosine: 0.004342
# key: head_6_3_v, cosine: 0.005205
# key: feedforward1_6_bias, cosine: 0.005260
# key: head_6_8_v, cosine: 0.005317
# key: head_6_6_v, cosine: 0.005449
# key: proj_5_weights, cosine: 0.006893
# key: head_5_11_v, cosine: 0.006944
# key: head_6_2_v, cosine: 0.007028
# key: head_5_7_v, cosine: 0.007152
# key: head_6_5_k, cosine: 0.007464
# key: head_5_2_k, cosine: 0.007586
# key: head_5_8_k, cosine: 0.007739
# key: head_6_2_q, cosine: 0.007793
# key: head_5_0_q, cosine: 0.008545
# key: head_5_8_q, cosine: 0.008762
# key: head_6_8_q, cosine: 0.008784
# key: feedforward1_5_bias, cosine: 0.008835
# key: head_6_6_k, cosine: 0.009041
# key: head_5_2_q, cosine: 0.009405
# key: head_5_4_v, cosine: 0.010645
# key: head_6_7_q, cosine: 0.013121
# key: head_5_0_k, cosine: 0.013199
# key: head_6_6_q, cosine: 0.014395
# key: head_5_11_k, cosine: 0.015091
# key: head_6_11_k, cosine: 0.015550
# key: head_6_10_q, cosine: 0.016431
# key: head_5_7_k, cosine: 0.017454
# key: head_6_0_q, cosine: 0.017743
# key: head_6_3_k, cosine: 0.017807
# key: head_5_7_q, cosine: 0.019357
# key: head_5_9_k, cosine: 0.020727
# key: head_6_4_q, cosine: 0.021380
# key: head_5_9_q, cosine: 0.023063
# key: head_5_8_v, cosine: 0.025757
# key: atten_norm_6_coeff, cosine: 0.027002
# key: proj_6_bias, cosine: 0.047485
# key: proj_5_bias, cosine: 0.062671
# key: norm_5_coeff, cosine: 0.080933
# key: norm_6_bias, cosine: 0.109258
# key: atten_norm_5_coeff, cosine: 0.113130
# key: norm_5_bias, cosine: 0.165394
# key: atten_norm_5_bias, cosine: 0.379400
# key: feedforward2_4_bias, cosine: 0.487998 (the first tensor that has nothing to do with the new layers)
# key: atten_norm_6_bias, cosine: 0.491329
# key: proj_4_bias, cosine: 0.557496
# key: feedforward1_4_bias, cosine: 0.593886
# key: feedforward2_5_bias, cosine: 0.649885
# key: proj_1_bias, cosine: 0.674996 (the first tensor that has nothing to do with the new layers and the layer before the new layers)
# key: atten_norm_4_bias, cosine: 0.709713
# key: feedforward2_6_bias, cosine: 0.775972 (the most similar tensor in the new layers)
# key: head_4_8_q, cosine: 0.808647
# key: atten_norm_2_bias, cosine: 0.852352
# key: feedforward2_3_bias, cosine: 0.867385
# token_id: 438, token: "┌", head: 4, cosine: 0.191362
# token_id: 416, token: "►", head: 10, cosine: 0.250198
# token_id: 416, token: "►", head: 4, cosine: 0.298854
# token_id: 17, token: "﬩", head: 2, cosine: 0.353135
# token_id: 438, token: "┌", head: 9, cosine: 0.397071
# token_id: 483, token: "░", head: 4, cosine: 0.416134
# token_id: 416, token: "►", head: 1, cosine: 0.439858
# token_id: 458, token: "🦀", head: 6, cosine: 0.444882
# token_id: 416, token: "►", head: 7, cosine: 0.448754
# token_id: 111, token: "├", head: 8, cosine: 0.451548
# token_id: 416, token: "►", head: 0, cosine: 0.458469
# token_id: 27, token: "·", head: 9, cosine: 0.477642
# token_id: 350, token: "˴", head: 0, cosine: 0.492794
# token_id: 341, token: "’", head: 7, cosine: 0.505220
# token_id: 458, token: "🦀", head: 9, cosine: 0.505254
# token_id: 438, token: "┌", head: 11, cosine: 0.509739
# token_id: 199, token: "┤", head: 3, cosine: 0.510461
# token_id: 27, token: "·", head: 4, cosine: 0.511308
# token_id: 147, token: "\u{a0}", head: 4, cosine: 0.514281
# token_id: 180, token: "✕", head: 10, cosine: 0.526506
# token_id: 199, token: "┤", head: 1, cosine: 0.535973
# token_id: 129, token: "﴿", head: 7, cosine: 0.536396
# token_id: 128, token: "ߺ", head: 8, cosine: 0.542053
# token_id: 114, token: "┬", head: 3, cosine: 0.544813
# token_id: 114, token: "┬", head: 7, cosine: 0.544952
# token_id: 338, token: "┗", head: 3, cosine: 0.551518
# token_id: 436, token: "╭", head: 3, cosine: 0.555989
# token_id: 180, token: "✕", head: 8, cosine: 0.564123
# token_id: 338, token: "┗", head: 7, cosine: 0.569463
# token_id: 114, token: "┬", head: 4, cosine: 0.571647
# token_id: 394, token: "┐", head: 9, cosine: 0.576385
# token_id: 438, token: "┌", head: 1, cosine: 0.578594
# token_id: 416, token: "►", head: 6, cosine: 0.580162
# token_id: 350, token: "˴", head: 8, cosine: 0.581571
# token_id: 114, token: "┬", head: 5, cosine: 0.587970
# token_id: 168, token: "→", head: 0, cosine: 0.589506
# token_id: 341, token: "’", head: 10, cosine: 0.591798
# token_id: 152, token: "│", head: 5, cosine: 0.592282
# token_id: 496, token: "▸", head: 3, cosine: 0.596853
# token_id: 458, token: "🦀", head: 8, cosine: 0.599263
# token_id: 338, token: "┗", head: 9, cosine: 0.600887
# token_id: 161, token: "━", head: 4, cosine: 0.602662
# token_id: 147, token: "\u{a0}", head: 2, cosine: 0.602853
# token_id: 416, token: "►", head: 5, cosine: 0.604358
# token_id: 111, token: "├", head: 7, cosine: 0.604926
# token_id: 426, token: "𐊛", head: 9, cosine: 0.618253
# token_id: 421, token: "²", head: 9, cosine: 0.619956
# token_id: 147, token: "\u{a0}", head: 9, cosine: 0.620302
# token_id: 199, token: "┤", head: 7, cosine: 0.621301
# token_id: 17, token: "﬩", head: 6, cosine: 0.621358
# token_id: 199, token: "┤", head: 8, cosine: 0.623913
# token_id: 375, token: "”", head: 7, cosine: 0.624708
# token_id: 350, token: "˴", head: 7, cosine: 0.624854
# token_id: 472, token: "\u{3000}", head: 1, cosine: 0.625906
# token_id: 375, token: "”", head: 5, cosine: 0.628780
# token_id: 375, token: "”", head: 6, cosine: 0.630891
# token_id: 350, token: "˴", head: 4, cosine: 0.632130
# token_id: 114, token: "┬", head: 2, cosine: 0.632956
# token_id: 114, token: "┬", head: 10, cosine: 0.633084
# token_id: 314, token: "－", head: 5, cosine: 0.634678
# token_id: 111, token: "├", head: 2, cosine: 0.637860
# token_id: 111, token: "├", head: 3, cosine: 0.638832
# token_id: 438, token: "┌", head: 0, cosine: 0.639777
# token_id: 270, token: "︖", head: 2, cosine: 0.641644
# token_id: 180, token: "✕", head: 5, cosine: 0.643196
# token_id: 68, token: "@", head: 5, cosine: 0.644114
# token_id: 338, token: "┗", head: 2, cosine: 0.644748
# token_id: 168, token: "→", head: 4, cosine: 0.644950
# token_id: 438, token: "┌", head: 3, cosine: 0.645267
# token_id: 199, token: "┤", head: 11, cosine: 0.646076
# token_id: 382, token: "—", head: 4, cosine: 0.649366
# token_id: 147, token: "\u{a0}", head: 6, cosine: 0.650232
# token_id: 469, token: "［", head: 10, cosine: 0.650319
# token_id: 496, token: "▸", head: 5, cosine: 0.651851
# token_id: 314, token: "－", head: 7, cosine: 0.651954
# token_id: 27, token: "·", head: 7, cosine: 0.652399
# token_id: 483, token: "░", head: 1, cosine: 0.653012
# token_id: 416, token: "►", head: 3, cosine: 0.653126
# token_id: 56, token: "┼", head: 8, cosine: 0.653573
# token_id: 93, token: "\u{2009}", head: 3, cosine: 0.654478
# token_id: 180, token: "✕", head: 0, cosine: 0.654925
# token_id: 339, token: "~", head: 8, cosine: 0.655272
# token_id: 339, token: "~", head: 3, cosine: 0.655823
# token_id: 65, token: "․", head: 7, cosine: 0.660752
# token_id: 382, token: "—", head: 5, cosine: 0.662099
# token_id: 338, token: "┗", head: 8, cosine: 0.662789
# token_id: 394, token: "┐", head: 3, cosine: 0.662951
# token_id: 147, token: "\u{a0}", head: 5, cosine: 0.662978
# token_id: 180, token: "✕", head: 1, cosine: 0.663876
# token_id: 367, token: "╴", head: 9, cosine: 0.665217
# token_id: 197, token: "\u{2007}", head: 1, cosine: 0.666686
# token_id: 493, token: "―", head: 5, cosine: 0.667286
# token_id: 111, token: "├", head: 0, cosine: 0.667298
# token_id: 180, token: "✕", head: 4, cosine: 0.668481
# token_id: 62, token: "“", head: 11, cosine: 0.669149
# token_id: 65, token: "․", head: 11, cosine: 0.670152
# token_id: 496, token: "▸", head: 10, cosine: 0.677048
# token_id: 416, token: "►", head: 2, cosine: 0.677506
# token_id: 394, token: "┐", head: 11, cosine: 0.677974
# token_id: 199, token: "┤", head: 5, cosine: 0.679385
#
# model-l7-2 vs model-l7-4 (share layer_0..=layer_4)
# key: atten_norm_6_coeff, cosine: -0.089869
# key: norm_6_coeff, cosine: -0.085242
# key: norm_6_bias, cosine: -0.076232
# key: feedforward1_6_bias, cosine: -0.048226
# key: head_5_3_q, cosine: -0.037262
# key: head_6_6_k, cosine: -0.034790
# key: head_6_0_k, cosine: -0.034119
# key: head_5_10_q, cosine: -0.025474
# key: head_6_9_v, cosine: -0.024700
# key: head_5_4_q, cosine: -0.023990
# key: head_6_2_k, cosine: -0.023795
# key: head_5_3_k, cosine: -0.023276
# key: head_6_2_v, cosine: -0.020841
# key: head_5_1_v, cosine: -0.020838
# key: head_5_10_k, cosine: -0.019973
# key: head_6_7_k, cosine: -0.019475
# key: head_6_0_q, cosine: -0.015434
# key: head_6_8_v, cosine: -0.015426
# key: head_5_4_k, cosine: -0.015151
# key: head_6_9_q, cosine: -0.014306
# key: head_6_2_q, cosine: -0.011589
# key: head_6_1_v, cosine: -0.011455
# key: head_6_5_v, cosine: -0.010970
# key: head_5_5_v, cosine: -0.010791
# key: head_5_1_k, cosine: -0.010491
# key: head_6_11_k, cosine: -0.010320
# key: head_5_6_k, cosine: -0.010052
# key: head_5_0_v, cosine: -0.010047
# key: head_6_11_q, cosine: -0.008861
# key: head_5_2_v, cosine: -0.008574
# key: head_6_3_v, cosine: -0.007155
# key: head_5_10_v, cosine: -0.006895
# key: head_5_11_q, cosine: -0.006463
# key: head_6_6_q, cosine: -0.006328
# key: head_5_6_q, cosine: -0.005300
# key: feedforward2_6_weights, cosine: -0.005238
# key: head_6_1_k, cosine: -0.004793
# key: head_6_7_q, cosine: -0.004731
# key: proj_6_weights, cosine: -0.004429
# key: head_5_5_k, cosine: -0.004191
# key: head_6_10_v, cosine: -0.002956
# key: head_5_1_q, cosine: -0.002680
# key: head_5_9_v, cosine: -0.002110
# key: head_6_11_v, cosine: -0.002080
# key: head_6_10_q, cosine: -0.001207
# key: head_6_1_q, cosine: -0.000992
# key: head_5_2_k, cosine: -0.000358
# key: head_6_5_k, cosine: -0.000348
# key: head_6_4_q, cosine: 0.000719
# key: feedforward2_5_weights, cosine: 0.001331
# key: feedforward1_5_weights, cosine: 0.001355
# key: head_5_8_q, cosine: 0.001608
# key: feedforward1_6_weights, cosine: 0.001971
# key: head_5_3_v, cosine: 0.002317
# key: head_6_8_q, cosine: 0.002891
# key: head_5_5_q, cosine: 0.002931
# key: head_5_6_v, cosine: 0.003167
# key: head_6_6_v, cosine: 0.003713
# key: head_6_8_k, cosine: 0.003862
# key: head_5_8_k, cosine: 0.003972
# key: head_6_7_v, cosine: 0.004416
# key: head_6_9_k, cosine: 0.004568
# key: head_6_4_v, cosine: 0.005502
# key: head_5_7_k, cosine: 0.005603
# key: head_6_0_v, cosine: 0.005877
# key: head_5_11_k, cosine: 0.006380
# key: head_5_11_v, cosine: 0.007294
# key: head_5_2_q, cosine: 0.007707
# key: proj_5_weights, cosine: 0.008224
# key: head_5_4_v, cosine: 0.008982
# key: head_5_9_k, cosine: 0.008990
# key: head_5_7_v, cosine: 0.009186
# key: head_5_0_k, cosine: 0.010556
# key: head_5_0_q, cosine: 0.010916
# key: head_6_4_k, cosine: 0.011874
# key: head_6_10_k, cosine: 0.012069
# key: head_5_7_q, cosine: 0.013858
# key: head_6_3_q, cosine: 0.014820
# key: head_6_3_k, cosine: 0.016357
# key: head_6_5_q, cosine: 0.017962
# key: head_5_8_v, cosine: 0.019204
# key: head_5_9_q, cosine: 0.026442
# key: proj_5_bias, cosine: 0.031243
# key: feedforward1_5_bias, cosine: 0.034718
# key: proj_6_bias, cosine: 0.050830
# key: norm_5_coeff, cosine: 0.098599
# key: norm_5_bias, cosine: 0.123271
# key: atten_norm_5_coeff, cosine: 0.124520
# key: atten_norm_5_bias, cosine: 0.345675
# key: feedforward2_4_bias, cosine: 0.462596 (the first tensor that has nothing to do with the new layers)
# key: atten_norm_6_bias, cosine: 0.511741
# key: proj_4_bias, cosine: 0.576389
# key: feedforward1_4_bias, cosine: 0.618695
# key: feedforward2_5_bias, cosine: 0.619286
# key: atten_norm_4_bias, cosine: 0.687602
# key: proj_1_bias, cosine: 0.703318 (the first tensor that has nothing to do with the new layers and the layer before the new layers)
# key: feedforward2_6_bias, cosine: 0.753383 (the most similar tensor in the new layers)
# key: head_4_8_q, cosine: 0.795709
# key: atten_norm_2_bias, cosine: 0.855419
# key: feedforward2_3_bias, cosine: 0.868776
# token_id: 111, token: "├", head: 8, cosine: 0.316450
# token_id: 17, token: "﬩", head: 2, cosine: 0.353135
# token_id: 438, token: "┌", head: 4, cosine: 0.391071
# token_id: 111, token: "├", head: 3, cosine: 0.415112
# token_id: 483, token: "░", head: 4, cosine: 0.416134
# token_id: 496, token: "▸", head: 3, cosine: 0.417142
# token_id: 438, token: "┌", head: 9, cosine: 0.429702
# token_id: 111, token: "├", head: 7, cosine: 0.435880
# token_id: 341, token: "’", head: 7, cosine: 0.450034
# token_id: 111, token: "├", head: 10, cosine: 0.450863
# token_id: 496, token: "▸", head: 10, cosine: 0.453813
# token_id: 180, token: "✕", head: 8, cosine: 0.463663
# token_id: 111, token: "├", head: 9, cosine: 0.489775
# token_id: 438, token: "┌", head: 3, cosine: 0.491433
# token_id: 350, token: "˴", head: 0, cosine: 0.492794
# token_id: 148, token: "╾", head: 6, cosine: 0.495577
# token_id: 199, token: "┤", head: 3, cosine: 0.505862
# token_id: 438, token: "┌", head: 11, cosine: 0.506048
# token_id: 111, token: "├", head: 4, cosine: 0.536329
# token_id: 129, token: "﴿", head: 7, cosine: 0.536396
# token_id: 148, token: "╾", head: 0, cosine: 0.540342
# token_id: 128, token: "ߺ", head: 8, cosine: 0.542053
# token_id: 375, token: "”", head: 5, cosine: 0.552659
# token_id: 496, token: "▸", head: 8, cosine: 0.557081
# token_id: 339, token: "~", head: 8, cosine: 0.570028
# token_id: 458, token: "🦀", head: 6, cosine: 0.574098
# token_id: 180, token: "✕", head: 0, cosine: 0.574128
# token_id: 148, token: "╾", head: 2, cosine: 0.576686
# token_id: 112, token: "⚬", head: 11, cosine: 0.579366
# token_id: 350, token: "˴", head: 8, cosine: 0.581571
# token_id: 199, token: "┤", head: 1, cosine: 0.583750
# token_id: 148, token: "╾", head: 10, cosine: 0.587768
# token_id: 339, token: "~", head: 5, cosine: 0.595762
# token_id: 496, token: "▸", head: 5, cosine: 0.597780
# token_id: 338, token: "┗", head: 3, cosine: 0.599939
# token_id: 339, token: "~", head: 1, cosine: 0.601027
# token_id: 111, token: "├", head: 0, cosine: 0.603153
# token_id: 180, token: "✕", head: 4, cosine: 0.604457
# token_id: 27, token: "·", head: 4, cosine: 0.604628
# token_id: 112, token: "⚬", head: 10, cosine: 0.605599
# token_id: 180, token: "✕", head: 1, cosine: 0.608349
# token_id: 199, token: "┤", head: 8, cosine: 0.609981
# token_id: 496, token: "▸", head: 2, cosine: 0.611912
# token_id: 496, token: "▸", head: 1, cosine: 0.612058
# token_id: 27, token: "·", head: 6, cosine: 0.614283
# token_id: 394, token: "┐", head: 9, cosine: 0.616279
# token_id: 68, token: "@", head: 9, cosine: 0.616555
# token_id: 27, token: "·", head: 9, cosine: 0.618231
# token_id: 426, token: "𐊛", head: 9, cosine: 0.618253
# token_id: 17, token: "﬩", head: 6, cosine: 0.621358
# token_id: 148, token: "╾", head: 3, cosine: 0.623287
# token_id: 350, token: "˴", head: 7, cosine: 0.624854
# token_id: 472, token: "\u{3000}", head: 1, cosine: 0.625906
# token_id: 339, token: "~", head: 10, cosine: 0.626381
# token_id: 199, token: "┤", head: 7, cosine: 0.631576
# token_id: 350, token: "˴", head: 4, cosine: 0.632130
# token_id: 338, token: "┗", head: 9, cosine: 0.632927
# token_id: 112, token: "⚬", head: 3, cosine: 0.633469
# token_id: 367, token: "╴", head: 9, cosine: 0.634326
# token_id: 382, token: "—", head: 4, cosine: 0.635724
# token_id: 147, token: "\u{a0}", head: 4, cosine: 0.638108
# token_id: 168, token: "→", head: 0, cosine: 0.638976
# token_id: 112, token: "⚬", head: 7, cosine: 0.640590
# token_id: 458, token: "🦀", head: 9, cosine: 0.641145
# token_id: 270, token: "︖", head: 2, cosine: 0.641644
# token_id: 148, token: "╾", head: 4, cosine: 0.643647
# token_id: 416, token: "►", head: 4, cosine: 0.645201
# token_id: 199, token: "┤", head: 4, cosine: 0.648194
# token_id: 469, token: "［", head: 10, cosine: 0.650319
# token_id: 199, token: "┤", head: 5, cosine: 0.650748
# token_id: 111, token: "├", head: 5, cosine: 0.650795
# token_id: 68, token: "@", head: 7, cosine: 0.651563
# token_id: 51, token: "┛", head: 1, cosine: 0.652177
# token_id: 483, token: "░", head: 1, cosine: 0.653012
# token_id: 56, token: "┼", head: 8, cosine: 0.653573
# token_id: 147, token: "\u{a0}", head: 11, cosine: 0.653627
# token_id: 93, token: "\u{2009}", head: 3, cosine: 0.654478
# token_id: 367, token: "╴", head: 4, cosine: 0.654758
# token_id: 355, token: "╿", head: 6, cosine: 0.657016
# token_id: 170, token: "\\", head: 10, cosine: 0.659658
# token_id: 199, token: "┤", head: 11, cosine: 0.659743
# token_id: 341, token: "’", head: 1, cosine: 0.661395
# token_id: 437, token: "タ", head: 5, cosine: 0.661965
# token_id: 375, token: "”", head: 6, cosine: 0.663904
# token_id: 180, token: "✕", head: 5, cosine: 0.664488
# token_id: 148, token: "╾", head: 11, cosine: 0.665016
# token_id: 197, token: "\u{2007}", head: 1, cosine: 0.666686
# token_id: 148, token: "╾", head: 1, cosine: 0.668550
# token_id: 382, token: "—", head: 5, cosine: 0.669642
# token_id: 112, token: "⚬", head: 9, cosine: 0.669854
# token_id: 341, token: "’", head: 10, cosine: 0.671781
# token_id: 112, token: "⚬", head: 6, cosine: 0.671792
# token_id: 68, token: "@", head: 5, cosine: 0.672682
# token_id: 496, token: "▸", head: 9, cosine: 0.673215
# token_id: 199, token: "┤", head: 6, cosine: 0.674043
# token_id: 112, token: "⚬", head: 0, cosine: 0.676348
# token_id: 438, token: "┌", head: 0, cosine: 0.677233
# token_id: 416, token: "►", head: 10, cosine: 0.680369
# token_id: 365, token: "∅", head: 1, cosine: 0.680741
# token_id: 180, token: "✕", head: 10, cosine: 0.680913
#
# model-l7-3 vs model-l7-4 (share layer_0..=layer_5)
# key: norm_6_coeff, cosine: -0.137397
# key: norm_6_bias, cosine: -0.070732
# key: head_6_11_v, cosine: -0.024637
# key: head_6_5_v, cosine: -0.020849
# key: head_6_0_q, cosine: -0.015682
# key: head_6_9_q, cosine: -0.010346
# key: head_6_8_q, cosine: -0.009438
# key: head_6_1_k, cosine: -0.009129
# key: head_6_6_v, cosine: -0.008784
# key: head_6_1_v, cosine: -0.007692
# key: head_6_11_q, cosine: -0.006744
# key: head_6_3_q, cosine: -0.006271
# key: head_6_10_k, cosine: -0.004719
# key: head_6_11_k, cosine: -0.004331
# key: head_6_7_q, cosine: -0.003535
# key: head_6_9_v, cosine: -0.002699
# key: head_6_10_q, cosine: -0.002385
# key: head_6_9_k, cosine: -0.002103
# key: head_6_2_v, cosine: -0.001589
# key: head_6_0_v, cosine: -0.001287
# key: head_6_8_k, cosine: -0.000923
# key: head_6_2_k, cosine: -0.000807
# key: feedforward1_6_weights, cosine: -0.000599
# key: head_6_6_k, cosine: 0.000021
# key: head_6_2_q, cosine: 0.001007
# key: head_6_10_v, cosine: 0.001746
# key: head_6_4_k, cosine: 0.002350
# key: feedforward2_6_weights, cosine: 0.002631
# key: head_6_4_q, cosine: 0.003937
# key: head_6_5_q, cosine: 0.004325
# key: head_6_5_k, cosine: 0.005070
# key: head_6_6_q, cosine: 0.006954
# key: head_6_4_v, cosine: 0.007682
# key: head_6_3_k, cosine: 0.008143
# key: proj_6_weights, cosine: 0.008767
# key: head_6_0_k, cosine: 0.010612
# key: head_6_3_v, cosine: 0.011225
# key: head_6_1_q, cosine: 0.013664
# key: head_6_8_v, cosine: 0.014344
# key: head_6_7_k, cosine: 0.023002
# key: head_6_7_v, cosine: 0.025267
# key: feedforward1_6_bias, cosine: 0.040159
# key: atten_norm_6_coeff, cosine: 0.058965
# key: proj_6_bias, cosine: 0.094884
# key: atten_norm_6_bias, cosine: 0.294460
# key: atten_norm_5_bias, cosine: 0.619295 (the first tensor that has nothing to do with the new layer)
# key: feedforward1_5_bias, cosine: 0.662252
# key: feedforward2_6_bias, cosine: 0.724775 (the most similar tensor in the new layer)
# key: proj_5_bias, cosine: 0.760960
# key: feedforward2_5_bias, cosine: 0.768618
# key: norm_5_bias, cosine: 0.822213
# key: feedforward2_4_bias, cosine: 0.845782 (the first tensor that has nothing to do with the new layer and the layer before the new layer)
# key: proj_4_bias, cosine: 0.862406
# key: head_5_9_k, cosine: 0.877207
# key: feedforward1_4_bias, cosine: 0.888157
# key: head_5_4_q, cosine: 0.894247
# key: head_5_4_k, cosine: 0.896058
# key: head_5_1_k, cosine: 0.904552
# key: atten_norm_4_bias, cosine: 0.904851
# key: head_5_11_k, cosine: 0.905506
# key: head_5_9_q, cosine: 0.907415
# key: head_5_0_k, cosine: 0.908727
# key: head_5_11_q, cosine: 0.909324
# key: head_5_8_q, cosine: 0.910260
# key: head_5_1_q, cosine: 0.910331
# key: head_5_7_k, cosine: 0.912843
# key: head_5_8_k, cosine: 0.915023
# key: head_5_5_q, cosine: 0.915626
# key: head_5_6_q, cosine: 0.919064
# key: head_5_3_q, cosine: 0.920047
# key: head_5_10_q, cosine: 0.920421
# key: head_5_6_k, cosine: 0.923143
# key: proj_1_bias, cosine: 0.923388
# key: head_5_2_k, cosine: 0.924666
# key: head_5_5_k, cosine: 0.927481
# key: head_5_0_q, cosine: 0.928239
# key: head_5_2_q, cosine: 0.931110
# key: head_5_10_k, cosine: 0.932467
# key: head_5_3_k, cosine: 0.933526
# key: head_5_7_q, cosine: 0.933628
# key: proj_5_weights, cosine: 0.935906
# key: feedforward2_5_weights, cosine: 0.944024
# key: atten_norm_2_bias, cosine: 0.947024
# key: feedforward1_1_weights, cosine: 0.950985
# key: head_5_5_v, cosine: 0.952586
# key: atten_norm_5_coeff, cosine: 0.954290
# key: feedforward2_4_weights, cosine: 0.954475
# key: head_5_9_v, cosine: 0.955538
# key: head_5_1_v, cosine: 0.956447
# key: head_5_0_v, cosine: 0.956632
# key: head_5_7_v, cosine: 0.956957
# key: head_5_8_v, cosine: 0.956984
# key: head_5_10_v, cosine: 0.957625
# key: head_5_3_v, cosine: 0.958087
# key: norm_5_coeff, cosine: 0.958430
# key: head_5_2_v, cosine: 0.959951
# key: head_5_11_v, cosine: 0.960944
# key: head_5_6_v, cosine: 0.961389
# key: head_5_4_v, cosine: 0.961521
# key: proj_1_weights, cosine: 0.962006
# token_id: 341, token: "’", head: 7, cosine: 0.445759
# token_id: 180, token: "✕", head: 8, cosine: 0.534599
# token_id: 111, token: "├", head: 8, cosine: 0.574172
# token_id: 112, token: "⚬", head: 11, cosine: 0.579366
# token_id: 180, token: "✕", head: 5, cosine: 0.583822
# token_id: 112, token: "⚬", head: 10, cosine: 0.605599
# token_id: 180, token: "✕", head: 1, cosine: 0.609719
# token_id: 421, token: "²", head: 9, cosine: 0.619956
# token_id: 58, token: "➖", head: 3, cosine: 0.633123
# token_id: 341, token: "’", head: 10, cosine: 0.633160
# token_id: 112, token: "⚬", head: 3, cosine: 0.633469
# token_id: 112, token: "⚬", head: 7, cosine: 0.640590
# token_id: 58, token: "➖", head: 0, cosine: 0.657578
# token_id: 148, token: "╾", head: 2, cosine: 0.663887
# token_id: 416, token: "►", head: 4, cosine: 0.666020
# token_id: 112, token: "⚬", head: 9, cosine: 0.669854
# token_id: 112, token: "⚬", head: 6, cosine: 0.671792
# token_id: 180, token: "✕", head: 0, cosine: 0.673894
# token_id: 112, token: "⚬", head: 0, cosine: 0.676348
# token_id: 341, token: "’", head: 6, cosine: 0.676875
# token_id: 367, token: "╴", head: 9, cosine: 0.686521
# token_id: 148, token: "╾", head: 6, cosine: 0.686616
# token_id: 112, token: "⚬", head: 2, cosine: 0.687285
# token_id: 340, token: "τ", head: 8, cosine: 0.687720
# token_id: 217, token: "−", head: 10, cosine: 0.689235
# token_id: 180, token: "✕", head: 4, cosine: 0.690306
# token_id: 58, token: "➖", head: 4, cosine: 0.691155
# token_id: 496, token: "▸", head: 8, cosine: 0.691773
# token_id: 180, token: "✕", head: 10, cosine: 0.693575
# token_id: 58, token: "➖", head: 8, cosine: 0.695444
# token_id: 363, token: "H", head: 4, cosine: 0.698386
# token_id: 58, token: "➖", head: 5, cosine: 0.706944
# token_id: 438, token: "┌", head: 9, cosine: 0.707819
# token_id: 179, token: "‘", head: 10, cosine: 0.708412
# token_id: 58, token: "➖", head: 6, cosine: 0.708476
# token_id: 340, token: "τ", head: 3, cosine: 0.710815
# token_id: 341, token: "’", head: 1, cosine: 0.723471
# token_id: 266, token: "╰", head: 1, cosine: 0.723990
# token_id: 111, token: "├", head: 7, cosine: 0.724426
# token_id: 148, token: "╾", head: 10, cosine: 0.726108
# token_id: 179, token: "‘", head: 5, cosine: 0.727943
# token_id: 421, token: "²", head: 11, cosine: 0.728511
# token_id: 112, token: "⚬", head: 8, cosine: 0.728599
# token_id: 206, token: "X", head: 3, cosine: 0.732427
# token_id: 367, token: "╴", head: 10, cosine: 0.734741
# token_id: 266, token: "╰", head: 3, cosine: 0.734960
# token_id: 136, token: "⚠", head: 1, cosine: 0.734977
# token_id: 340, token: "τ", head: 7, cosine: 0.735793
# token_id: 340, token: "τ", head: 2, cosine: 0.742952
# token_id: 180, token: "✕", head: 3, cosine: 0.743874
# token_id: 340, token: "τ", head: 1, cosine: 0.743948
# token_id: 148, token: "╾", head: 3, cosine: 0.744272
# token_id: 179, token: "‘", head: 4, cosine: 0.744642
# token_id: 405, token: "‶", head: 10, cosine: 0.744773
# token_id: 180, token: "✕", head: 9, cosine: 0.746161
# token_id: 496, token: "▸", head: 10, cosine: 0.747155
# token_id: 179, token: "‘", head: 2, cosine: 0.747613
# token_id: 436, token: "╭", head: 3, cosine: 0.747659
# token_id: 136, token: "⚠", head: 6, cosine: 0.748075
# token_id: 405, token: "‶", head: 7, cosine: 0.751040
# token_id: 148, token: "╾", head: 0, cosine: 0.751492
# token_id: 416, token: "►", head: 1, cosine: 0.751798
# token_id: 148, token: "╾", head: 11, cosine: 0.751914
# token_id: 179, token: "‘", head: 7, cosine: 0.755880
# token_id: 111, token: "├", head: 0, cosine: 0.756289
# token_id: 179, token: "‘", head: 1, cosine: 0.756610
# token_id: 421, token: "²", head: 4, cosine: 0.757688
# token_id: 65, token: "․", head: 7, cosine: 0.758679
# token_id: 27, token: "·", head: 4, cosine: 0.758752
# token_id: 69, token: "А", head: 8, cosine: 0.759134
# token_id: 25, token: "）", head: 7, cosine: 0.759587
# token_id: 179, token: "‘", head: 9, cosine: 0.759672
# token_id: 496, token: "▸", head: 4, cosine: 0.760816
# token_id: 65, token: "․", head: 11, cosine: 0.764407
# token_id: 340, token: "τ", head: 11, cosine: 0.765728
# token_id: 331, token: "″", head: 5, cosine: 0.766624
# token_id: 180, token: "✕", head: 6, cosine: 0.766654
# token_id: 180, token: "✕", head: 7, cosine: 0.767230
# token_id: 114, token: "┬", head: 5, cosine: 0.768602
# token_id: 383, token: "ー", head: 1, cosine: 0.769342
# token_id: 188, token: "L", head: 7, cosine: 0.770348
# token_id: 148, token: "╾", head: 4, cosine: 0.771658
# token_id: 359, token: "�", head: 7, cosine: 0.772860
# token_id: 90, token: "܂", head: 10, cosine: 0.773088
# token_id: 170, token: "\\", head: 11, cosine: 0.774732
# token_id: 16, token: "˗", head: 9, cosine: 0.774745
# token_id: 359, token: "�", head: 1, cosine: 0.775124
# token_id: 112, token: "⚬", head: 1, cosine: 0.775450
# token_id: 266, token: "╰", head: 0, cosine: 0.776881
# token_id: 416, token: "►", head: 10, cosine: 0.779365
# token_id: 25, token: "）", head: 5, cosine: 0.779431
# token_id: 147, token: "\u{a0}", head: 2, cosine: 0.780079
# token_id: 367, token: "╴", head: 4, cosine: 0.780223
# token_id: 276, token: "Ⲻ", head: 2, cosine: 0.780438
# token_id: 136, token: "⚠", head: 10, cosine: 0.781115
# token_id: 0, token: "‟", head: 4, cosine: 0.781264
# token_id: 58, token: "➖", head: 7, cosine: 0.781327
# token_id: 180, token: "✕", head: 11, cosine: 0.783910
# token_id: 161, token: "━", head: 4, cosine: 0.784075
# token_id: 58, token: "➖", head: 2, cosine: 0.784952

# let's continue with l7-1
cp model-l7-1.dat model-l7.dat;

# It's using too much memory, so I cannot run this in daytime. I'll just run batches and come back tomorrow.
cargo run --release -- insert-layer --input model-l7.dat --output model-l8-1.dat --insert-at 7;
cargo run --release -- insert-layer --input model-l7.dat --output model-l8-2.dat --insert-at 7;

for _ in 0..8 {
    cargo run --release -- train --model model-l8-1.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l8-2.dat --steps 31;
    sleep 200sec;
}

cargo run --release -- insert-layer --input model-l8-1.dat --output model-l9-1.dat --insert-at 8;
cargo run --release -- insert-layer --input model-l8-1.dat --output model-l9-2.dat --insert-at 8;
cargo run --release -- insert-layer --input model-l8-2.dat --output model-l9-3.dat --insert-at 8;
cargo run --release -- insert-layer --input model-l8-2.dat --output model-l9-4.dat --insert-at 8;

# NOTE: a step takes roughly 25 ~ 35 seconds on M3 Pro
# NOTE: It has to be `_ in 0..8`, but I didn't have enough time so I Ctrl+C before the last iteration.
for _ in 0..7 {
    cargo run --release -- train --model model-l9-1.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-2.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-3.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-4.dat --steps 31;
    sleep 200sec;
}

# model losses (last 3)
# model-l9-1: 2.059, 2.335, 2.052
# model-l9-2: 2.163, 2.148, 2.014
# model-l9-3: 2.034, 2.132, 2.225
# model-l9-4: 2.096, 2.150, 2.173

cargo run --release -- compare model-l9-1.dat model-l9-2.dat --limit 100;
# share layer_0..=layer_7
# key: proj_8_bias, cosine: -0.146473
# key: feedforward1_8_bias, cosine: -0.054183
# key: head_8_1_v, cosine: -0.030292
# key: head_8_4_q, cosine: -0.022117
# key: head_8_10_v, cosine: -0.015885
# key: head_8_6_v, cosine: -0.014373
# key: head_8_9_k, cosine: -0.011131
# key: head_8_1_k, cosine: -0.010526
# key: head_8_5_v, cosine: -0.009948
# key: head_8_2_k, cosine: -0.009266
# key: head_8_7_k, cosine: -0.009048
# key: head_8_4_v, cosine: -0.008160
# key: head_8_3_k, cosine: -0.008031
# key: head_8_5_q, cosine: -0.006316
# key: head_8_6_k, cosine: -0.005149
# key: head_8_7_v, cosine: -0.005065
# key: head_8_8_q, cosine: -0.002201
# key: feedforward1_8_weights, cosine: -0.002171
# key: head_8_8_v, cosine: -0.001822
# key: head_8_10_q, cosine: -0.001407
# key: proj_8_weights, cosine: -0.001308
# key: head_8_3_q, cosine: -0.001158
# key: feedforward2_8_weights, cosine: -0.001108
# key: head_8_2_q, cosine: -0.001055
# key: head_8_10_k, cosine: 0.000550
# key: head_8_0_q, cosine: 0.003923
# key: head_8_4_k, cosine: 0.003959
# key: head_8_3_v, cosine: 0.004236
# key: head_8_9_v, cosine: 0.004847
# key: head_8_11_k, cosine: 0.005961
# key: head_8_11_q, cosine: 0.006200
# key: head_8_0_k, cosine: 0.006620
# key: head_8_0_v, cosine: 0.006704
# key: norm_8_bias, cosine: 0.006814
# key: head_8_7_q, cosine: 0.006933
# key: head_8_5_k, cosine: 0.008112
# key: head_8_6_q, cosine: 0.008706
# key: head_8_8_k, cosine: 0.008872
# key: head_8_1_q, cosine: 0.011886
# key: head_8_11_v, cosine: 0.013595
# key: atten_norm_8_coeff, cosine: 0.018291
# key: head_8_2_v, cosine: 0.021012
# key: head_8_9_q, cosine: 0.028003
# key: norm_8_coeff, cosine: 0.051010
# key: atten_norm_8_bias, cosine: 0.284512
# key: feedforward2_8_bias, cosine: 0.506225
# key: proj_7_bias, cosine: 0.682306
# key: feedforward1_6_bias, cosine: 0.687991
# key: atten_norm_7_bias, cosine: 0.689236
# key: feedforward1_5_bias, cosine: 0.696567
# key: feedforward1_7_bias, cosine: 0.698319
# key: norm_7_bias, cosine: 0.771400
# key: feedforward2_7_bias, cosine: 0.773454
# key: atten_norm_6_bias, cosine: 0.798288
# key: feedforward2_5_bias, cosine: 0.828782
# key: atten_norm_5_bias, cosine: 0.835805
# key: head_7_0_q, cosine: 0.873610
# key: norm_6_bias, cosine: 0.879760
# key: feedforward2_6_bias, cosine: 0.882145
# key: head_7_7_q, cosine: 0.887339
# key: proj_6_bias, cosine: 0.892503
# key: head_7_9_q, cosine: 0.898596
# key: head_7_1_q, cosine: 0.905631
# key: head_7_0_k, cosine: 0.908006
# key: head_7_10_q, cosine: 0.909263
# key: head_7_9_k, cosine: 0.909412
# key: head_7_5_q, cosine: 0.910131
# key: head_7_8_q, cosine: 0.910600
# key: head_7_11_k, cosine: 0.913260
# key: head_7_11_q, cosine: 0.913439
# key: head_7_6_k, cosine: 0.915259
# key: head_7_3_k, cosine: 0.918368
# key: head_7_7_k, cosine: 0.918584
# key: head_7_2_q, cosine: 0.919738
# key: head_7_6_q, cosine: 0.919892
# key: head_7_3_q, cosine: 0.920921
# key: head_7_1_k, cosine: 0.925114
# key: head_7_8_k, cosine: 0.926153
# key: proj_5_bias, cosine: 0.931111
# key: head_7_2_k, cosine: 0.931549
# key: norm_5_bias, cosine: 0.933958
# key: head_7_5_k, cosine: 0.935098
# key: feedforward2_7_weights, cosine: 0.936624
# key: proj_7_weights, cosine: 0.939144
# key: head_7_10_k, cosine: 0.943191
# key: atten_norm_7_coeff, cosine: 0.946095
# key: proj_4_bias, cosine: 0.946220
# key: head_7_4_k, cosine: 0.946932
# key: head_7_4_q, cosine: 0.950195
# key: norm_7_coeff, cosine: 0.952264
# key: feedforward2_4_bias, cosine: 0.953830
# key: head_6_10_q, cosine: 0.954462
# key: head_7_5_v, cosine: 0.955294
# key: head_7_1_v, cosine: 0.955582
# key: head_7_6_v, cosine: 0.956726
# key: head_7_10_v, cosine: 0.956815
# key: head_7_4_v, cosine: 0.957312
# key: head_6_7_k, cosine: 0.957326
# key: proj_6_weights, cosine: 0.957357
# key: feedforward2_6_weights, cosine: 0.957427
# token_id: 202, token: "㇓", head: 4, cosine: 0.490262
# token_id: 106, token: "Ⳇ", head: 8, cosine: 0.492350
# token_id: 202, token: "㇓", head: 7, cosine: 0.616648
# token_id: 202, token: "㇓", head: 10, cosine: 0.627962
# token_id: 420, token: "⁃", head: 0, cosine: 0.642357
# token_id: 202, token: "㇓", head: 9, cosine: 0.642853
# token_id: 339, token: "~", head: 9, cosine: 0.649328
# token_id: 510, token: "、", head: 8, cosine: 0.665311
# token_id: 202, token: "㇓", head: 6, cosine: 0.677703
# token_id: 27, token: "·", head: 9, cosine: 0.687249
# token_id: 202, token: "㇓", head: 3, cosine: 0.700788
# token_id: 106, token: "Ⳇ", head: 6, cosine: 0.702637
# token_id: 202, token: "㇓", head: 5, cosine: 0.703557
# token_id: 202, token: "㇓", head: 1, cosine: 0.711443
# token_id: 249, token: "ノ", head: 7, cosine: 0.715932
# token_id: 237, token: "┘", head: 11, cosine: 0.722762
# token_id: 420, token: "⁃", head: 3, cosine: 0.723021
# token_id: 202, token: "㇓", head: 8, cosine: 0.725005
# token_id: 27, token: "·", head: 7, cosine: 0.726284
# token_id: 136, token: "⚠", head: 6, cosine: 0.728533
# token_id: 496, token: "▸", head: 3, cosine: 0.732131
# token_id: 106, token: "Ⳇ", head: 0, cosine: 0.737926
# token_id: 77, token: "┯", head: 11, cosine: 0.740001
# token_id: 106, token: "Ⳇ", head: 4, cosine: 0.740119
# token_id: 510, token: "、", head: 0, cosine: 0.741669
# token_id: 496, token: "▸", head: 1, cosine: 0.743487
# token_id: 202, token: "㇓", head: 0, cosine: 0.745848
# token_id: 249, token: "ノ", head: 9, cosine: 0.749165
# token_id: 27, token: "·", head: 5, cosine: 0.750719
# token_id: 136, token: "⚠", head: 3, cosine: 0.754531
# token_id: 106, token: "Ⳇ", head: 3, cosine: 0.757060
# token_id: 400, token: "└", head: 11, cosine: 0.758379
# token_id: 420, token: "⁃", head: 5, cosine: 0.763827
# token_id: 68, token: "@", head: 3, cosine: 0.764955
# token_id: 106, token: "Ⳇ", head: 7, cosine: 0.767027
# token_id: 400, token: "└", head: 9, cosine: 0.770900
# token_id: 416, token: "►", head: 2, cosine: 0.771473
# token_id: 114, token: "┬", head: 0, cosine: 0.773981
# token_id: 119, token: "⊃", head: 11, cosine: 0.774415
# token_id: 114, token: "┬", head: 8, cosine: 0.774502
# token_id: 27, token: "·", head: 4, cosine: 0.775451
# token_id: 77, token: "┯", head: 0, cosine: 0.775571
# token_id: 217, token: "−", head: 10, cosine: 0.777839
# token_id: 420, token: "⁃", head: 7, cosine: 0.779633
# token_id: 106, token: "Ⳇ", head: 5, cosine: 0.781894
# token_id: 159, token: "™", head: 8, cosine: 0.782112
# token_id: 106, token: "Ⳇ", head: 10, cosine: 0.782451
# token_id: 249, token: "ノ", head: 2, cosine: 0.782901
# token_id: 400, token: "└", head: 3, cosine: 0.783040
# token_id: 420, token: "⁃", head: 1, cosine: 0.784333
# token_id: 510, token: "、", head: 2, cosine: 0.785552
# token_id: 27, token: "·", head: 1, cosine: 0.786295
# token_id: 198, token: "﹘", head: 5, cosine: 0.788077
# token_id: 202, token: "㇓", head: 2, cosine: 0.788418
# token_id: 106, token: "Ⳇ", head: 9, cosine: 0.788533
# token_id: 505, token: "‒", head: 0, cosine: 0.789309
# token_id: 237, token: "┘", head: 9, cosine: 0.791297
# token_id: 237, token: "┘", head: 7, cosine: 0.792933
# token_id: 106, token: "Ⳇ", head: 2, cosine: 0.793784
# token_id: 119, token: "⊃", head: 3, cosine: 0.795944
# token_id: 114, token: "┬", head: 2, cosine: 0.795998
# token_id: 29, token: "─", head: 11, cosine: 0.796510
# token_id: 202, token: "㇓", head: 11, cosine: 0.796942
# token_id: 438, token: "┌", head: 7, cosine: 0.797691
# token_id: 159, token: "™", head: 11, cosine: 0.798518
# token_id: 77, token: "┯", head: 1, cosine: 0.799265
# token_id: 29, token: "─", head: 4, cosine: 0.802034
# token_id: 276, token: "Ⲻ", head: 6, cosine: 0.803550
# token_id: 383, token: "ー", head: 8, cosine: 0.807811
# token_id: 136, token: "⚠", head: 1, cosine: 0.809000
# token_id: 416, token: "►", head: 10, cosine: 0.810362
# token_id: 332, token: "⧸", head: 5, cosine: 0.811573
# token_id: 383, token: "ー", head: 1, cosine: 0.811762
# token_id: 219, token: "丿", head: 4, cosine: 0.811828
# token_id: 114, token: "┬", head: 7, cosine: 0.813392
# token_id: 505, token: "‒", head: 11, cosine: 0.813692
# token_id: 198, token: "﹘", head: 6, cosine: 0.813752
# token_id: 30, token: "〳", head: 10, cosine: 0.814053
# token_id: 77, token: "┯", head: 4, cosine: 0.814151
# token_id: 237, token: "┘", head: 6, cosine: 0.814167
# token_id: 119, token: "⊃", head: 9, cosine: 0.814282
# token_id: 332, token: "⧸", head: 7, cosine: 0.814888
# token_id: 276, token: "Ⲻ", head: 2, cosine: 0.817617
# token_id: 436, token: "╭", head: 9, cosine: 0.818149
# token_id: 237, token: "┘", head: 1, cosine: 0.818269
# token_id: 27, token: "·", head: 2, cosine: 0.819084
# token_id: 49, token: "۔", head: 4, cosine: 0.819159
# token_id: 303, token: "%", head: 6, cosine: 0.819505
# token_id: 119, token: "⊃", head: 1, cosine: 0.821011
# token_id: 198, token: "﹘", head: 10, cosine: 0.821043
# token_id: 420, token: "⁃", head: 9, cosine: 0.821205
# token_id: 136, token: "⚠", head: 11, cosine: 0.821237
# token_id: 436, token: "╭", head: 8, cosine: 0.823168
# token_id: 249, token: "ノ", head: 0, cosine: 0.823881
# token_id: 237, token: "┘", head: 3, cosine: 0.824853
# token_id: 339, token: "~", head: 11, cosine: 0.825529
# token_id: 119, token: "⊃", head: 4, cosine: 0.825731
# token_id: 505, token: "‒", head: 1, cosine: 0.826247
# token_id: 510, token: "、", head: 3, cosine: 0.826790
# token_id: 170, token: "\\", head: 10, cosine: 0.827261

cargo run --release -- compare model-l9-1.dat model-l9-3.dat --limit 100;
# share layer_0..=layer_6
# key: norm_8_bias, cosine: -0.074359
# key: head_7_1_k, cosine: -0.031946
# key: atten_norm_8_coeff, cosine: -0.028345
# key: norm_7_bias, cosine: -0.026255
# key: feedforward1_7_bias, cosine: -0.022972
# key: head_7_4_v, cosine: -0.022847
# key: head_7_8_q, cosine: -0.022635
# key: head_8_8_q, cosine: -0.021669
# key: head_8_10_k, cosine: -0.019454
# key: head_8_1_v, cosine: -0.017496
# key: head_8_0_v, cosine: -0.015338
# key: head_8_10_v, cosine: -0.015316
# key: head_7_10_v, cosine: -0.014855
# key: head_7_6_k, cosine: -0.010360
# key: head_7_2_k, cosine: -0.009477
# key: head_7_7_v, cosine: -0.008780
# key: head_7_7_k, cosine: -0.007589
# key: head_8_2_k, cosine: -0.007558
# key: head_8_5_v, cosine: -0.007348
# key: head_8_6_k, cosine: -0.006719
# key: head_7_11_v, cosine: -0.006602
# key: feedforward1_8_bias, cosine: -0.006358
# key: head_7_0_v, cosine: -0.005979
# key: head_8_11_k, cosine: -0.005912
# key: head_7_10_k, cosine: -0.005280
# key: head_7_6_v, cosine: -0.005267
# key: proj_8_bias, cosine: -0.005143
# key: head_7_5_k, cosine: -0.004895
# key: head_8_2_q, cosine: -0.004039
# key: head_8_3_q, cosine: -0.002774
# key: head_8_2_v, cosine: -0.002600
# key: feedforward1_8_weights, cosine: -0.002409
# key: head_7_3_q, cosine: -0.002336
# key: head_8_7_k, cosine: -0.002219
# key: feedforward1_7_weights, cosine: -0.002124
# key: head_7_2_q, cosine: -0.001700
# key: feedforward2_7_weights, cosine: -0.001282
# key: head_8_7_v, cosine: -0.001072
# key: head_7_5_q, cosine: -0.001068
# key: head_8_4_q, cosine: -0.000515
# key: head_7_0_k, cosine: -0.000108
# key: head_7_7_q, cosine: 0.000210
# key: feedforward2_8_weights, cosine: 0.000623
# key: head_7_5_v, cosine: 0.000695
# key: head_8_6_q, cosine: 0.000798
# key: head_8_5_q, cosine: 0.000845
# key: head_7_9_k, cosine: 0.000880
# key: head_7_4_q, cosine: 0.000908
# key: head_7_11_q, cosine: 0.001207
# key: head_7_9_q, cosine: 0.001825
# key: head_7_3_v, cosine: 0.001894
# key: head_8_3_k, cosine: 0.002048
# key: proj_8_weights, cosine: 0.002993
# key: head_7_9_v, cosine: 0.003649
# key: head_8_1_k, cosine: 0.003693
# key: head_7_10_q, cosine: 0.003700
# key: head_8_7_q, cosine: 0.004056
# key: head_7_8_v, cosine: 0.004089
# key: head_7_3_k, cosine: 0.004276
# key: head_8_9_k, cosine: 0.004679
# key: head_7_0_q, cosine: 0.005107
# key: head_7_11_k, cosine: 0.005276
# key: norm_7_coeff, cosine: 0.006233
# key: head_8_10_q, cosine: 0.006289
# key: proj_7_weights, cosine: 0.007062
# key: head_8_8_v, cosine: 0.007745
# key: head_7_8_k, cosine: 0.007987
# key: head_8_6_v, cosine: 0.008151
# key: head_8_11_q, cosine: 0.008864
# key: head_7_4_k, cosine: 0.009669
# key: head_8_3_v, cosine: 0.009811
# key: head_8_8_k, cosine: 0.010576
# key: head_8_1_q, cosine: 0.010587
# key: head_8_4_k, cosine: 0.012321
# key: head_8_4_v, cosine: 0.012603
# key: head_7_1_v, cosine: 0.012987
# key: head_7_6_q, cosine: 0.013173
# key: head_8_5_k, cosine: 0.015163
# key: head_8_0_k, cosine: 0.019694
# key: head_8_9_v, cosine: 0.020590
# key: head_8_9_q, cosine: 0.020662
# key: head_8_0_q, cosine: 0.022252
# key: head_7_2_v, cosine: 0.024291
# key: head_8_11_v, cosine: 0.026535
# key: atten_norm_7_coeff, cosine: 0.029035
# key: head_7_1_q, cosine: 0.031610
# key: norm_8_coeff, cosine: 0.072374
# key: proj_7_bias, cosine: 0.130183
# key: atten_norm_7_bias, cosine: 0.136079
# key: atten_norm_8_bias, cosine: 0.198885
# key: feedforward2_8_bias, cosine: 0.351578
# key: feedforward1_6_bias, cosine: 0.446362
# key: feedforward1_5_bias, cosine: 0.492607
# key: feedforward2_7_bias, cosine: 0.497242
# key: atten_norm_6_bias, cosine: 0.542633
# key: atten_norm_5_bias, cosine: 0.663493
# key: feedforward2_6_bias, cosine: 0.675980
# key: head_6_10_q, cosine: 0.708241
# key: head_6_9_q, cosine: 0.711442
# key: norm_6_bias, cosine: 0.728620
# token_id: 199, token: "┤", head: 1, cosine: 0.212688
# token_id: 27, token: "·", head: 7, cosine: 0.301342
# token_id: 199, token: "┤", head: 3, cosine: 0.317228
# token_id: 114, token: "┬", head: 5, cosine: 0.326069
# token_id: 199, token: "┤", head: 4, cosine: 0.330520
# token_id: 267, token: "ï", head: 2, cosine: 0.346274
# token_id: 114, token: "┬", head: 3, cosine: 0.355140
# token_id: 111, token: "├", head: 0, cosine: 0.370798
# token_id: 114, token: "┬", head: 7, cosine: 0.394114
# token_id: 266, token: "╰", head: 2, cosine: 0.427661
# token_id: 27, token: "·", head: 9, cosine: 0.430715
# token_id: 341, token: "’", head: 8, cosine: 0.441980
# token_id: 267, token: "ï", head: 10, cosine: 0.448044
# token_id: 27, token: "·", head: 5, cosine: 0.469763
# token_id: 267, token: "ï", head: 5, cosine: 0.477729
# token_id: 114, token: "┬", head: 9, cosine: 0.491175
# token_id: 267, token: "ï", head: 1, cosine: 0.492011
# token_id: 148, token: "╾", head: 2, cosine: 0.492144
# token_id: 341, token: "’", head: 6, cosine: 0.492850
# token_id: 199, token: "┤", head: 5, cosine: 0.507238
# token_id: 27, token: "·", head: 10, cosine: 0.522121
# token_id: 199, token: "┤", head: 9, cosine: 0.523905
# token_id: 267, token: "ï", head: 0, cosine: 0.526266
# token_id: 254, token: "€", head: 2, cosine: 0.529430
# token_id: 148, token: "╾", head: 0, cosine: 0.533440
# token_id: 89, token: "ß", head: 3, cosine: 0.536677
# token_id: 27, token: "·", head: 4, cosine: 0.537180
# token_id: 341, token: "’", head: 7, cosine: 0.545816
# token_id: 114, token: "┬", head: 2, cosine: 0.550699
# token_id: 209, token: "┃", head: 4, cosine: 0.550775
# token_id: 372, token: "┙", head: 11, cosine: 0.551652
# token_id: 114, token: "┬", head: 6, cosine: 0.552504
# token_id: 111, token: "├", head: 7, cosine: 0.553072
# token_id: 114, token: "┬", head: 8, cosine: 0.553863
# token_id: 457, token: "Ü", head: 0, cosine: 0.558626
# token_id: 199, token: "┤", head: 6, cosine: 0.561752
# token_id: 27, token: "·", head: 2, cosine: 0.567368
# token_id: 375, token: "”", head: 5, cosine: 0.570053
# token_id: 180, token: "✕", head: 4, cosine: 0.576108
# token_id: 367, token: "╴", head: 9, cosine: 0.576991
# token_id: 114, token: "┬", head: 4, cosine: 0.582226
# token_id: 148, token: "╾", head: 5, cosine: 0.586514
# token_id: 457, token: "Ü", head: 11, cosine: 0.586855
# token_id: 267, token: "ï", head: 11, cosine: 0.589046
# token_id: 114, token: "┬", head: 11, cosine: 0.591609
# token_id: 267, token: "ï", head: 3, cosine: 0.594569
# token_id: 266, token: "╰", head: 6, cosine: 0.596823
# token_id: 199, token: "┤", head: 10, cosine: 0.597998
# token_id: 199, token: "┤", head: 7, cosine: 0.598081
# token_id: 294, token: "Z", head: 3, cosine: 0.601734
# token_id: 209, token: "┃", head: 3, cosine: 0.605430
# token_id: 209, token: "┃", head: 10, cosine: 0.605577
# token_id: 180, token: "✕", head: 0, cosine: 0.605919
# token_id: 199, token: "┤", head: 8, cosine: 0.607221
# token_id: 341, token: "’", head: 10, cosine: 0.607415
# token_id: 438, token: "┌", head: 3, cosine: 0.610650
# token_id: 129, token: "﴿", head: 7, cosine: 0.612366
# token_id: 27, token: "·", head: 1, cosine: 0.612632
# token_id: 199, token: "┤", head: 0, cosine: 0.617741
# token_id: 209, token: "┃", head: 11, cosine: 0.618340
# token_id: 254, token: "€", head: 1, cosine: 0.620926
# token_id: 341, token: "’", head: 5, cosine: 0.624175
# token_id: 343, token: "🌸", head: 8, cosine: 0.627255
# token_id: 341, token: "’", head: 0, cosine: 0.632050
# token_id: 341, token: "’", head: 1, cosine: 0.634205
# token_id: 148, token: "╾", head: 3, cosine: 0.634482
# token_id: 148, token: "╾", head: 1, cosine: 0.635858
# token_id: 305, token: "µ", head: 1, cosine: 0.637244
# token_id: 114, token: "┬", head: 0, cosine: 0.637390
# token_id: 267, token: "ï", head: 4, cosine: 0.639828
# token_id: 341, token: "’", head: 9, cosine: 0.639857
# token_id: 209, token: "┃", head: 6, cosine: 0.640422
# token_id: 209, token: "┃", head: 0, cosine: 0.642572
# token_id: 180, token: "✕", head: 1, cosine: 0.644225
# token_id: 266, token: "╰", head: 1, cosine: 0.644359
# token_id: 111, token: "├", head: 5, cosine: 0.644767
# token_id: 148, token: "╾", head: 6, cosine: 0.645887
# token_id: 127, token: "ö", head: 4, cosine: 0.647249
# token_id: 209, token: "┃", head: 9, cosine: 0.647709
# token_id: 209, token: "┃", head: 5, cosine: 0.647943
# token_id: 111, token: "├", head: 10, cosine: 0.649423
# token_id: 27, token: "·", head: 0, cosine: 0.651732
# token_id: 254, token: "€", head: 8, cosine: 0.653814
# token_id: 33, token: "和", head: 0, cosine: 0.655825
# token_id: 457, token: "Ü", head: 7, cosine: 0.656881
# token_id: 111, token: "├", head: 3, cosine: 0.658369
# token_id: 111, token: "├", head: 1, cosine: 0.658855
# token_id: 100, token: "Ö", head: 4, cosine: 0.658887
# token_id: 266, token: "╰", head: 8, cosine: 0.661542
# token_id: 294, token: "Z", head: 7, cosine: 0.662302
# token_id: 127, token: "ö", head: 0, cosine: 0.664360
# token_id: 254, token: "€", head: 5, cosine: 0.665820
# token_id: 438, token: "┌", head: 4, cosine: 0.667206
# token_id: 27, token: "·", head: 6, cosine: 0.668465
# token_id: 148, token: "╾", head: 11, cosine: 0.668934
# token_id: 148, token: "╾", head: 4, cosine: 0.670402
# token_id: 209, token: "┃", head: 1, cosine: 0.670989
# token_id: 496, token: "▸", head: 8, cosine: 0.671440
# token_id: 267, token: "ï", head: 7, cosine: 0.671838
# token_id: 129, token: "﴿", head: 8, cosine: 0.672414

cargo run --release -- compare model-l9-1.dat model-l9-4.dat --limit 100;
# share layer_0..=layer_6
# key: norm_8_bias, cosine: -0.109985
# key: norm_7_bias, cosine: -0.053509
# key: head_7_1_k, cosine: -0.024090
# key: head_7_4_v, cosine: -0.022650
# key: proj_7_bias, cosine: -0.020517
# key: head_7_8_q, cosine: -0.019946
# key: head_8_2_k, cosine: -0.018468
# key: head_8_8_k, cosine: -0.017322
# key: head_8_2_v, cosine: -0.016368
# key: atten_norm_8_coeff, cosine: -0.014895
# key: head_8_1_v, cosine: -0.014300
# key: head_7_7_k, cosine: -0.013854
# key: head_8_6_v, cosine: -0.013657
# key: head_8_1_q, cosine: -0.013577
# key: head_8_11_v, cosine: -0.013179
# key: head_7_5_k, cosine: -0.012895
# key: head_8_0_v, cosine: -0.011214
# key: head_7_6_k, cosine: -0.011074
# key: head_8_10_q, cosine: -0.009656
# key: head_8_7_v, cosine: -0.009464
# key: head_7_10_v, cosine: -0.009160
# key: head_8_9_q, cosine: -0.008832
# key: head_8_5_k, cosine: -0.008736
# key: head_7_2_k, cosine: -0.008495
# key: head_8_8_q, cosine: -0.008293
# key: head_7_11_q, cosine: -0.007117
# key: head_8_1_k, cosine: -0.007079
# key: head_8_3_q, cosine: -0.006694
# key: head_8_10_k, cosine: -0.006186
# key: head_8_4_q, cosine: -0.005867
# key: head_8_0_q, cosine: -0.005851
# key: head_8_8_v, cosine: -0.005262
# key: proj_8_weights, cosine: -0.004490
# key: head_7_2_q, cosine: -0.003998
# key: head_8_7_q, cosine: -0.003705
# key: head_7_9_q, cosine: -0.003573
# key: head_7_7_v, cosine: -0.003466
# key: head_7_0_v, cosine: -0.003437
# key: head_8_4_v, cosine: -0.002970
# key: head_8_9_v, cosine: -0.002563
# key: head_7_6_v, cosine: -0.002181
# key: feedforward1_8_weights, cosine: -0.001774
# key: feedforward1_7_weights, cosine: -0.001744
# key: head_7_11_v, cosine: -0.001432
# key: feedforward2_7_weights, cosine: -0.001069
# key: feedforward1_7_bias, cosine: -0.000997
# key: head_7_0_k, cosine: -0.000827
# key: head_7_9_k, cosine: -0.000642
# key: head_7_10_k, cosine: 0.000289
# key: head_8_3_k, cosine: 0.000514
# key: head_7_0_q, cosine: 0.001050
# key: head_7_3_v, cosine: 0.001306
# key: head_8_11_k, cosine: 0.001390
# key: head_8_9_k, cosine: 0.001404
# key: feedforward2_8_weights, cosine: 0.002033
# key: head_7_5_q, cosine: 0.002096
# key: head_7_11_k, cosine: 0.002109
# key: head_7_8_k, cosine: 0.002938
# key: head_8_4_k, cosine: 0.003818
# key: head_7_3_k, cosine: 0.003888
# key: head_7_10_q, cosine: 0.003929
# key: head_8_6_q, cosine: 0.005502
# key: head_7_7_q, cosine: 0.005832
# key: head_7_5_v, cosine: 0.006018
# key: head_7_3_q, cosine: 0.006279
# key: head_8_11_q, cosine: 0.006505
# key: head_8_3_v, cosine: 0.006845
# key: head_8_6_k, cosine: 0.007580
# key: head_7_8_v, cosine: 0.007841
# key: proj_7_weights, cosine: 0.007912
# key: head_7_4_q, cosine: 0.007914
# key: head_8_10_v, cosine: 0.008794
# key: head_7_6_q, cosine: 0.009067
# key: atten_norm_7_coeff, cosine: 0.009293
# key: head_8_2_q, cosine: 0.009906
# key: head_7_9_v, cosine: 0.010006
# key: head_7_1_v, cosine: 0.011272
# key: norm_8_coeff, cosine: 0.013333
# key: head_7_4_k, cosine: 0.015307
# key: head_7_1_q, cosine: 0.017812
# key: head_8_5_v, cosine: 0.021291
# key: head_8_5_q, cosine: 0.021653
# key: head_7_2_v, cosine: 0.023408
# key: head_8_7_k, cosine: 0.025376
# key: head_8_0_k, cosine: 0.027292
# key: norm_7_coeff, cosine: 0.039174
# key: feedforward1_8_bias, cosine: 0.044875
# key: proj_8_bias, cosine: 0.068712
# key: atten_norm_7_bias, cosine: 0.253110
# key: atten_norm_8_bias, cosine: 0.287293
# key: feedforward2_8_bias, cosine: 0.500372
# key: feedforward1_6_bias, cosine: 0.507600
# key: atten_norm_6_bias, cosine: 0.531659
# key: feedforward2_7_bias, cosine: 0.567617
# key: proj_4_bias, cosine: 0.626987
# key: feedforward1_5_bias, cosine: 0.637083
# key: atten_norm_5_bias, cosine: 0.651782
# key: feedforward2_5_bias, cosine: 0.688971
# key: head_6_9_q, cosine: 0.701839
# key: head_6_7_k, cosine: 0.710336
# token_id: 27, token: "·", head: 7, cosine: 0.301342
# token_id: 237, token: "┘", head: 7, cosine: 0.350874
# token_id: 114, token: "┬", head: 3, cosine: 0.372456
# token_id: 416, token: "►", head: 2, cosine: 0.373435
# token_id: 114, token: "┬", head: 5, cosine: 0.417283
# token_id: 27, token: "·", head: 9, cosine: 0.430715
# token_id: 438, token: "┌", head: 3, cosine: 0.436427
# token_id: 199, token: "┤", head: 3, cosine: 0.439872
# token_id: 266, token: "╰", head: 2, cosine: 0.449210
# token_id: 27, token: "·", head: 5, cosine: 0.469763
# token_id: 199, token: "┤", head: 5, cosine: 0.491473
# token_id: 209, token: "┃", head: 4, cosine: 0.492081
# token_id: 148, token: "╾", head: 2, cosine: 0.492144
# token_id: 209, token: "┃", head: 0, cosine: 0.495431
# token_id: 199, token: "┤", head: 4, cosine: 0.506861
# token_id: 510, token: "、", head: 8, cosine: 0.517130
# token_id: 367, token: "╴", head: 9, cosine: 0.520353
# token_id: 27, token: "·", head: 10, cosine: 0.522121
# token_id: 199, token: "┤", head: 1, cosine: 0.527531
# token_id: 254, token: "€", head: 2, cosine: 0.529430
# token_id: 148, token: "╾", head: 0, cosine: 0.533440
# token_id: 89, token: "ß", head: 3, cosine: 0.536677
# token_id: 27, token: "·", head: 4, cosine: 0.537180
# token_id: 372, token: "┙", head: 11, cosine: 0.551652
# token_id: 457, token: "Ü", head: 0, cosine: 0.558626
# token_id: 367, token: "╴", head: 2, cosine: 0.565421
# token_id: 209, token: "┃", head: 6, cosine: 0.565578
# token_id: 27, token: "·", head: 2, cosine: 0.567368
# token_id: 209, token: "┃", head: 3, cosine: 0.576639
# token_id: 114, token: "┬", head: 6, cosine: 0.578916
# token_id: 341, token: "’", head: 3, cosine: 0.579505
# token_id: 148, token: "╾", head: 5, cosine: 0.586514
# token_id: 457, token: "Ü", head: 11, cosine: 0.586855
# token_id: 114, token: "┬", head: 7, cosine: 0.588743
# token_id: 375, token: "”", head: 5, cosine: 0.593328
# token_id: 237, token: "┘", head: 4, cosine: 0.593986
# token_id: 199, token: "┤", head: 8, cosine: 0.598428
# token_id: 209, token: "┃", head: 1, cosine: 0.603304
# token_id: 209, token: "┃", head: 10, cosine: 0.603553
# token_id: 294, token: "Z", head: 3, cosine: 0.606914
# token_id: 27, token: "·", head: 1, cosine: 0.612632
# token_id: 341, token: "’", head: 7, cosine: 0.618090
# token_id: 254, token: "€", head: 1, cosine: 0.620926
# token_id: 114, token: "┬", head: 8, cosine: 0.623299
# token_id: 199, token: "┤", head: 9, cosine: 0.623307
# token_id: 438, token: "┌", head: 8, cosine: 0.624657
# token_id: 343, token: "🌸", head: 8, cosine: 0.627255
# token_id: 237, token: "┘", head: 9, cosine: 0.628315
# token_id: 438, token: "┌", head: 9, cosine: 0.629272
# token_id: 338, token: "┗", head: 9, cosine: 0.632162
# token_id: 114, token: "┬", head: 0, cosine: 0.634011
# token_id: 341, token: "’", head: 1, cosine: 0.634433
# token_id: 148, token: "╾", head: 3, cosine: 0.634482
# token_id: 148, token: "╾", head: 1, cosine: 0.635858
# token_id: 199, token: "┤", head: 7, cosine: 0.636993
# token_id: 305, token: "µ", head: 1, cosine: 0.637244
# token_id: 266, token: "╰", head: 6, cosine: 0.641240
# token_id: 237, token: "┘", head: 8, cosine: 0.643802
# token_id: 209, token: "┃", head: 5, cosine: 0.644865
# token_id: 148, token: "╾", head: 6, cosine: 0.645887
# token_id: 341, token: "’", head: 9, cosine: 0.649702
# token_id: 127, token: "ö", head: 4, cosine: 0.650858
# token_id: 209, token: "┃", head: 9, cosine: 0.651107
# token_id: 27, token: "·", head: 0, cosine: 0.651732
# token_id: 199, token: "┤", head: 2, cosine: 0.653807
# token_id: 254, token: "€", head: 8, cosine: 0.653814
# token_id: 457, token: "Ü", head: 7, cosine: 0.656881
# token_id: 352, token: "∧", head: 2, cosine: 0.656967
# token_id: 237, token: "┘", head: 11, cosine: 0.657698
# token_id: 510, token: "、", head: 0, cosine: 0.658078
# token_id: 100, token: "Ö", head: 4, cosine: 0.658887
# token_id: 341, token: "’", head: 11, cosine: 0.659833
# token_id: 416, token: "►", head: 10, cosine: 0.660656
# token_id: 111, token: "├", head: 1, cosine: 0.663312
# token_id: 254, token: "€", head: 5, cosine: 0.665820
# token_id: 29, token: "─", head: 4, cosine: 0.668009
# token_id: 27, token: "·", head: 6, cosine: 0.668465
# token_id: 204, token: "∨", head: 11, cosine: 0.668643
# token_id: 266, token: "╰", head: 1, cosine: 0.668745
# token_id: 148, token: "╾", head: 11, cosine: 0.668934
# token_id: 127, token: "ö", head: 0, cosine: 0.669622
# token_id: 199, token: "┤", head: 0, cosine: 0.669671
# token_id: 148, token: "╾", head: 4, cosine: 0.670402
# token_id: 355, token: "╿", head: 7, cosine: 0.673225
# token_id: 148, token: "╾", head: 10, cosine: 0.676028
# token_id: 237, token: "┘", head: 3, cosine: 0.677216
# token_id: 305, token: "µ", head: 5, cosine: 0.677594
# token_id: 341, token: "’", head: 8, cosine: 0.678318
# token_id: 343, token: "🌸", head: 0, cosine: 0.680242
# token_id: 496, token: "▸", head: 1, cosine: 0.680898
# token_id: 204, token: "∨", head: 1, cosine: 0.682378
# token_id: 111, token: "├", head: 0, cosine: 0.683076
# token_id: 341, token: "’", head: 10, cosine: 0.683222
# token_id: 510, token: "、", head: 7, cosine: 0.684933
# token_id: 372, token: "┙", head: 8, cosine: 0.685659
# token_id: 209, token: "┃", head: 2, cosine: 0.686817
# token_id: 341, token: "’", head: 5, cosine: 0.691847
# token_id: 26, token: "┥", head: 0, cosine: 0.691903
# token_id: 209, token: "┃", head: 11, cosine: 0.692210
# token_id: 343, token: "🌸", head: 3, cosine: 0.692360

cargo run --release -- compare model-l9-2.dat model-l9-3.dat --limit 100;
# share layer_0..=layer_6
# key: feedforward1_8_bias, cosine: -0.055728
# key: head_8_11_k, cosine: -0.028827
# key: head_7_1_k, cosine: -0.028556
# key: norm_7_bias, cosine: -0.025853
# key: head_7_8_q, cosine: -0.024066
# key: head_7_4_v, cosine: -0.023214
# key: norm_8_coeff, cosine: -0.015925
# key: head_8_3_k, cosine: -0.015512
# key: head_8_1_k, cosine: -0.015468
# key: head_8_8_q, cosine: -0.014300
# key: head_7_10_v, cosine: -0.014263
# key: head_8_7_q, cosine: -0.012575
# key: head_8_6_k, cosine: -0.011684
# key: head_8_7_k, cosine: -0.011500
# key: head_7_7_v, cosine: -0.011065
# key: head_7_0_k, cosine: -0.009809
# key: head_7_11_v, cosine: -0.009573
# key: head_8_5_k, cosine: -0.009465
# key: head_8_0_q, cosine: -0.009027
# key: head_7_6_v, cosine: -0.008911
# key: head_7_0_v, cosine: -0.008861
# key: head_8_10_q, cosine: -0.008733
# key: head_8_0_v, cosine: -0.008576
# key: head_8_10_k, cosine: -0.006932
# key: head_7_6_k, cosine: -0.006544
# key: head_8_6_q, cosine: -0.006434
# key: head_8_5_v, cosine: -0.004694
# key: head_7_7_q, cosine: -0.004658
# key: head_8_3_v, cosine: -0.004157
# key: head_7_7_k, cosine: -0.003433
# key: head_7_2_k, cosine: -0.002507
# key: head_8_0_k, cosine: -0.002490
# key: head_7_5_v, cosine: -0.002042
# key: feedforward1_7_weights, cosine: -0.001759
# key: head_8_9_v, cosine: -0.001479
# key: feedforward2_7_weights, cosine: -0.001329
# key: feedforward1_8_weights, cosine: -0.001300
# key: head_8_2_v, cosine: -0.001103
# key: feedforward2_8_weights, cosine: -0.001022
# key: head_8_6_v, cosine: -0.000824
# key: head_8_1_v, cosine: -0.000471
# key: head_7_4_q, cosine: 0.000152
# key: head_7_10_k, cosine: 0.000170
# key: head_7_11_q, cosine: 0.000590
# key: head_7_3_k, cosine: 0.000698
# key: head_8_11_q, cosine: 0.000756
# key: head_7_0_q, cosine: 0.000888
# key: head_7_9_v, cosine: 0.001019
# key: head_7_10_q, cosine: 0.001303
# key: head_7_3_v, cosine: 0.001739
# key: norm_7_coeff, cosine: 0.001775
# key: head_7_11_k, cosine: 0.002103
# key: proj_8_weights, cosine: 0.002123
# key: head_7_5_q, cosine: 0.002194
# key: head_8_4_v, cosine: 0.002295
# key: head_7_9_q, cosine: 0.002763
# key: head_8_3_q, cosine: 0.003523
# key: head_7_9_k, cosine: 0.003654
# key: head_7_3_q, cosine: 0.003681
# key: head_8_11_v, cosine: 0.003844
# key: head_8_2_k, cosine: 0.004664
# key: head_7_8_k, cosine: 0.005252
# key: head_8_10_v, cosine: 0.006019
# key: head_7_2_q, cosine: 0.006434
# key: proj_7_weights, cosine: 0.007004
# key: head_7_1_v, cosine: 0.007108
# key: head_7_5_k, cosine: 0.007369
# key: head_8_5_q, cosine: 0.007630
# key: feedforward1_7_bias, cosine: 0.007935
# key: head_8_2_q, cosine: 0.009455
# key: head_8_4_q, cosine: 0.009563
# key: head_7_6_q, cosine: 0.010538
# key: head_8_1_q, cosine: 0.011151
# key: head_7_8_v, cosine: 0.011594
# key: head_8_4_k, cosine: 0.013383
# key: head_7_4_k, cosine: 0.015815
# key: head_8_9_k, cosine: 0.015942
# key: head_8_7_v, cosine: 0.019637
# key: head_8_8_k, cosine: 0.020040
# key: head_8_8_v, cosine: 0.022489
# key: head_7_2_v, cosine: 0.022528
# key: head_7_1_q, cosine: 0.029521
# key: head_8_9_q, cosine: 0.031047
# key: atten_norm_7_coeff, cosine: 0.046244
# key: norm_8_bias, cosine: 0.060335
# key: atten_norm_7_bias, cosine: 0.062303
# key: proj_8_bias, cosine: 0.080355
# key: proj_7_bias, cosine: 0.104415
# key: atten_norm_8_coeff, cosine: 0.138726
# key: atten_norm_8_bias, cosine: 0.355580
# key: feedforward2_7_bias, cosine: 0.395476
# key: feedforward2_8_bias, cosine: 0.436677
# key: feedforward1_5_bias, cosine: 0.463231
# key: atten_norm_6_bias, cosine: 0.477203
# key: feedforward1_6_bias, cosine: 0.485079
# key: atten_norm_5_bias, cosine: 0.645648
# key: feedforward2_6_bias, cosine: 0.648668
# key: head_6_9_q, cosine: 0.695168
# key: feedforward2_5_bias, cosine: 0.704568
# key: head_6_10_q, cosine: 0.709822
# token_id: 114, token: "┬", head: 5, cosine: 0.312947
# token_id: 267, token: "ï", head: 2, cosine: 0.329608
# token_id: 114, token: "┬", head: 7, cosine: 0.349585
# token_id: 77, token: "┯", head: 11, cosine: 0.395433
# token_id: 209, token: "┃", head: 4, cosine: 0.427944
# token_id: 114, token: "┬", head: 8, cosine: 0.434268
# token_id: 266, token: "╰", head: 2, cosine: 0.439235
# token_id: 267, token: "ï", head: 10, cosine: 0.450928
# token_id: 267, token: "ï", head: 5, cosine: 0.458933
# token_id: 114, token: "┬", head: 2, cosine: 0.485330
# token_id: 202, token: "㇓", head: 4, cosine: 0.490262
# token_id: 267, token: "ï", head: 1, cosine: 0.491300
# token_id: 148, token: "╾", head: 2, cosine: 0.492144
# token_id: 106, token: "Ⳇ", head: 8, cosine: 0.492350
# token_id: 209, token: "┃", head: 3, cosine: 0.496194
# token_id: 114, token: "┬", head: 9, cosine: 0.501628
# token_id: 199, token: "┤", head: 4, cosine: 0.503356
# token_id: 341, token: "’", head: 7, cosine: 0.504070
# token_id: 267, token: "ï", head: 0, cosine: 0.516561
# token_id: 114, token: "┬", head: 3, cosine: 0.517183
# token_id: 27, token: "·", head: 9, cosine: 0.524113
# token_id: 375, token: "”", head: 5, cosine: 0.526374
# token_id: 254, token: "€", head: 2, cosine: 0.529430
# token_id: 148, token: "╾", head: 0, cosine: 0.533440
# token_id: 209, token: "┃", head: 6, cosine: 0.534945
# token_id: 89, token: "ß", head: 3, cosine: 0.536677
# token_id: 114, token: "┬", head: 11, cosine: 0.544440
# token_id: 266, token: "╰", head: 3, cosine: 0.549563
# token_id: 372, token: "┙", head: 11, cosine: 0.551652
# token_id: 161, token: "━", head: 4, cosine: 0.554134
# token_id: 77, token: "┯", head: 0, cosine: 0.555140
# token_id: 114, token: "┬", head: 1, cosine: 0.555542
# token_id: 457, token: "Ü", head: 0, cosine: 0.558626
# token_id: 114, token: "┬", head: 4, cosine: 0.568200
# token_id: 180, token: "✕", head: 4, cosine: 0.576108
# token_id: 209, token: "┃", head: 5, cosine: 0.580915
# token_id: 209, token: "┃", head: 0, cosine: 0.581579
# token_id: 267, token: "ï", head: 3, cosine: 0.582366
# token_id: 266, token: "╰", head: 0, cosine: 0.583294
# token_id: 267, token: "ï", head: 11, cosine: 0.584345
# token_id: 148, token: "╾", head: 5, cosine: 0.586514
# token_id: 457, token: "Ü", head: 11, cosine: 0.586855
# token_id: 199, token: "┤", head: 3, cosine: 0.591300
# token_id: 77, token: "┯", head: 2, cosine: 0.595080
# token_id: 209, token: "┃", head: 11, cosine: 0.598646
# token_id: 209, token: "┃", head: 10, cosine: 0.605432
# token_id: 180, token: "✕", head: 0, cosine: 0.605919
# token_id: 111, token: "├", head: 0, cosine: 0.606670
# token_id: 199, token: "┤", head: 1, cosine: 0.609202
# token_id: 129, token: "﴿", head: 7, cosine: 0.612366
# token_id: 209, token: "┃", head: 2, cosine: 0.612652
# token_id: 209, token: "┃", head: 1, cosine: 0.614040
# token_id: 114, token: "┬", head: 0, cosine: 0.615693
# token_id: 202, token: "㇓", head: 7, cosine: 0.616648
# token_id: 254, token: "€", head: 1, cosine: 0.620926
# token_id: 27, token: "·", head: 4, cosine: 0.624692
# token_id: 343, token: "🌸", head: 8, cosine: 0.627255
# token_id: 51, token: "┛", head: 9, cosine: 0.627925
# token_id: 202, token: "㇓", head: 10, cosine: 0.627962
# token_id: 438, token: "┌", head: 4, cosine: 0.632434
# token_id: 148, token: "╾", head: 3, cosine: 0.634482
# token_id: 267, token: "ï", head: 4, cosine: 0.634512
# token_id: 148, token: "╾", head: 1, cosine: 0.635858
# token_id: 77, token: "┯", head: 7, cosine: 0.637223
# token_id: 305, token: "µ", head: 1, cosine: 0.637244
# token_id: 420, token: "⁃", head: 0, cosine: 0.642357
# token_id: 202, token: "㇓", head: 9, cosine: 0.642853
# token_id: 127, token: "ö", head: 4, cosine: 0.642915
# token_id: 180, token: "✕", head: 1, cosine: 0.644225
# token_id: 341, token: "’", head: 8, cosine: 0.644495
# token_id: 83, token: "▼", head: 3, cosine: 0.645033
# token_id: 148, token: "╾", head: 6, cosine: 0.645887
# token_id: 77, token: "┯", head: 5, cosine: 0.649377
# token_id: 266, token: "╰", head: 6, cosine: 0.651795
# token_id: 394, token: "┐", head: 11, cosine: 0.652274
# token_id: 111, token: "├", head: 3, cosine: 0.652311
# token_id: 77, token: "┯", head: 4, cosine: 0.652354
# token_id: 77, token: "┯", head: 8, cosine: 0.653671
# token_id: 254, token: "€", head: 8, cosine: 0.653814
# token_id: 367, token: "╴", head: 2, cosine: 0.654507
# token_id: 33, token: "和", head: 0, cosine: 0.655825
# token_id: 457, token: "Ü", head: 7, cosine: 0.656881
# token_id: 438, token: "┌", head: 3, cosine: 0.658037
# token_id: 266, token: "╰", head: 1, cosine: 0.658597
# token_id: 100, token: "Ö", head: 4, cosine: 0.658887
# token_id: 127, token: "ö", head: 0, cosine: 0.659674
# token_id: 267, token: "ï", head: 7, cosine: 0.663494
# token_id: 77, token: "┯", head: 1, cosine: 0.663739
# token_id: 27, token: "·", head: 2, cosine: 0.665503
# token_id: 199, token: "┤", head: 5, cosine: 0.665591
# token_id: 254, token: "€", head: 5, cosine: 0.665820
# token_id: 367, token: "╴", head: 9, cosine: 0.668788
# token_id: 148, token: "╾", head: 11, cosine: 0.668934
# token_id: 148, token: "╾", head: 4, cosine: 0.670402
# token_id: 436, token: "╭", head: 8, cosine: 0.671261
# token_id: 394, token: "┐", head: 3, cosine: 0.672180
# token_id: 129, token: "﴿", head: 8, cosine: 0.672414
# token_id: 438, token: "┌", head: 10, cosine: 0.672833
# token_id: 148, token: "╾", head: 10, cosine: 0.676028
# token_id: 339, token: "~", head: 9, cosine: 0.676295

cargo run --release -- compare model-l9-2.dat model-l9-4.dat --limit 100;
# share layer_0..=layer_6
# key: norm_7_bias, cosine: -0.055437
# key: head_8_11_v, cosine: -0.031701
# key: norm_8_bias, cosine: -0.030273
# key: head_8_5_k, cosine: -0.024711
# key: feedforward1_8_bias, cosine: -0.023948
# key: head_7_8_q, cosine: -0.023643
# key: head_7_4_v, cosine: -0.021248
# key: head_8_9_v, cosine: -0.021072
# key: head_7_1_k, cosine: -0.020983
# key: head_8_0_k, cosine: -0.017871
# key: head_8_3_v, cosine: -0.016973
# key: head_8_9_q, cosine: -0.014557
# key: head_8_8_k, cosine: -0.012895
# key: head_8_10_v, cosine: -0.011822
# key: head_8_11_k, cosine: -0.011691
# key: head_7_7_k, cosine: -0.009440
# key: head_7_0_k, cosine: -0.008795
# key: head_7_10_v, cosine: -0.008681
# key: head_7_0_v, cosine: -0.007341
# key: proj_8_weights, cosine: -0.006871
# key: head_8_9_k, cosine: -0.006744
# key: head_7_6_k, cosine: -0.006640
# key: head_7_6_v, cosine: -0.006494
# key: head_7_7_v, cosine: -0.005570
# key: head_8_8_v, cosine: -0.005546
# key: head_8_7_v, cosine: -0.005499
# key: head_7_11_q, cosine: -0.005364
# key: head_7_11_v, cosine: -0.004570
# key: head_8_4_k, cosine: -0.004248
# key: head_7_9_q, cosine: -0.004212
# key: head_8_5_q, cosine: -0.004049
# key: head_8_6_k, cosine: -0.003376
# key: head_8_4_q, cosine: -0.002989
# key: head_7_5_k, cosine: -0.002186
# key: feedforward1_7_weights, cosine: -0.001613
# key: head_8_3_q, cosine: -0.001538
# key: feedforward2_7_weights, cosine: -0.001411
# key: head_7_9_k, cosine: -0.000907
# key: head_7_2_k, cosine: -0.000757
# key: head_7_11_k, cosine: -0.000306
# key: feedforward1_8_weights, cosine: -0.000003
# key: head_8_2_v, cosine: 0.000253
# key: head_8_10_q, cosine: 0.000729
# key: head_7_10_q, cosine: 0.001134
# key: head_7_8_k, cosine: 0.001202
# key: head_8_7_k, cosine: 0.001374
# key: feedforward2_8_weights, cosine: 0.001557
# key: head_7_3_v, cosine: 0.001822
# key: head_7_3_k, cosine: 0.002232
# key: head_8_6_v, cosine: 0.002603
# key: head_7_7_q, cosine: 0.002767
# key: head_7_0_q, cosine: 0.002897
# key: head_7_5_v, cosine: 0.003018
# key: head_8_3_k, cosine: 0.003040
# key: head_7_5_q, cosine: 0.003176
# key: head_8_4_v, cosine: 0.003264
# key: head_8_10_k, cosine: 0.003265
# key: head_7_2_q, cosine: 0.003480
# key: proj_7_bias, cosine: 0.003656
# key: proj_8_bias, cosine: 0.004448
# key: head_8_7_q, cosine: 0.004486
# key: head_7_1_v, cosine: 0.005224
# key: head_7_9_v, cosine: 0.005642
# key: head_7_10_k, cosine: 0.006419
# key: proj_7_weights, cosine: 0.006876
# key: head_7_6_q, cosine: 0.007774
# key: head_8_1_k, cosine: 0.008245
# key: head_7_3_q, cosine: 0.008803
# key: head_7_4_q, cosine: 0.009037
# key: head_8_5_v, cosine: 0.009073
# key: head_8_1_q, cosine: 0.010431
# key: head_8_11_q, cosine: 0.011646
# key: head_8_2_q, cosine: 0.011727
# key: head_8_0_q, cosine: 0.014111
# key: head_7_8_v, cosine: 0.014144
# key: atten_norm_7_coeff, cosine: 0.016553
# key: head_7_1_q, cosine: 0.016601
# key: head_8_0_v, cosine: 0.016835
# key: feedforward1_7_bias, cosine: 0.017420
# key: head_8_8_q, cosine: 0.018284
# key: head_8_1_v, cosine: 0.018842
# key: head_8_2_k, cosine: 0.019212
# key: head_7_2_v, cosine: 0.020503
# key: head_7_4_k, cosine: 0.021029
# key: head_8_6_q, cosine: 0.025840
# key: norm_7_coeff, cosine: 0.034543
# key: norm_8_coeff, cosine: 0.051768
# key: atten_norm_8_coeff, cosine: 0.072149
# key: atten_norm_7_bias, cosine: 0.146208
# key: atten_norm_8_bias, cosine: 0.271601
# key: feedforward2_7_bias, cosine: 0.455572
# key: feedforward2_8_bias, cosine: 0.489142
# key: atten_norm_6_bias, cosine: 0.515132
# key: feedforward1_5_bias, cosine: 0.533166
# key: feedforward1_6_bias, cosine: 0.545307
# key: atten_norm_5_bias, cosine: 0.623489
# key: proj_6_bias, cosine: 0.647996
# key: head_6_9_q, cosine: 0.686595
# key: head_6_10_q, cosine: 0.701319
# key: feedforward2_6_bias, cosine: 0.706384
# token_id: 209, token: "┃", head: 4, cosine: 0.383159
# token_id: 77, token: "┯", head: 11, cosine: 0.395433
# token_id: 438, token: "┌", head: 3, cosine: 0.425217
# token_id: 114, token: "┬", head: 5, cosine: 0.437022
# token_id: 209, token: "┃", head: 3, cosine: 0.445089
# token_id: 209, token: "┃", head: 0, cosine: 0.447058
# token_id: 209, token: "┃", head: 6, cosine: 0.465428
# token_id: 438, token: "┌", head: 8, cosine: 0.484510
# token_id: 202, token: "㇓", head: 4, cosine: 0.490262
# token_id: 148, token: "╾", head: 2, cosine: 0.492144
# token_id: 106, token: "Ⳇ", head: 8, cosine: 0.492350
# token_id: 367, token: "╴", head: 2, cosine: 0.511892
# token_id: 209, token: "┃", head: 2, cosine: 0.523607
# token_id: 27, token: "·", head: 9, cosine: 0.524113
# token_id: 254, token: "€", head: 2, cosine: 0.529430
# token_id: 148, token: "╾", head: 0, cosine: 0.533440
# token_id: 89, token: "ß", head: 3, cosine: 0.536677
# token_id: 367, token: "╴", head: 9, cosine: 0.539986
# token_id: 209, token: "┃", head: 1, cosine: 0.541272
# token_id: 114, token: "┬", head: 8, cosine: 0.547348
# token_id: 375, token: "”", head: 5, cosine: 0.549286
# token_id: 372, token: "┙", head: 11, cosine: 0.551652
# token_id: 77, token: "┯", head: 0, cosine: 0.555140
# token_id: 266, token: "╰", head: 2, cosine: 0.558382
# token_id: 457, token: "Ü", head: 0, cosine: 0.558626
# token_id: 114, token: "┬", head: 7, cosine: 0.565437
# token_id: 266, token: "╰", head: 0, cosine: 0.570021
# token_id: 237, token: "┘", head: 7, cosine: 0.576195
# token_id: 438, token: "┌", head: 9, cosine: 0.577004
# token_id: 148, token: "╾", head: 5, cosine: 0.586514
# token_id: 457, token: "Ü", head: 11, cosine: 0.586855
# token_id: 209, token: "┃", head: 5, cosine: 0.590352
# token_id: 77, token: "┯", head: 2, cosine: 0.595080
# token_id: 114, token: "┬", head: 3, cosine: 0.595347
# token_id: 209, token: "┃", head: 10, cosine: 0.606274
# token_id: 341, token: "’", head: 3, cosine: 0.612627
# token_id: 341, token: "’", head: 7, cosine: 0.614309
# token_id: 202, token: "㇓", head: 7, cosine: 0.616648
# token_id: 338, token: "┗", head: 9, cosine: 0.616741
# token_id: 254, token: "€", head: 1, cosine: 0.620926
# token_id: 438, token: "┌", head: 4, cosine: 0.622570
# token_id: 27, token: "·", head: 4, cosine: 0.624692
# token_id: 29, token: "─", head: 4, cosine: 0.625156
# token_id: 343, token: "🌸", head: 8, cosine: 0.627255
# token_id: 202, token: "㇓", head: 10, cosine: 0.627962
# token_id: 510, token: "、", head: 7, cosine: 0.628841
# token_id: 114, token: "┬", head: 0, cosine: 0.630767
# token_id: 51, token: "┛", head: 9, cosine: 0.632236
# token_id: 114, token: "┬", head: 11, cosine: 0.633814
# token_id: 148, token: "╾", head: 3, cosine: 0.634482
# token_id: 148, token: "╾", head: 1, cosine: 0.635858
# token_id: 77, token: "┯", head: 7, cosine: 0.637223
# token_id: 305, token: "µ", head: 1, cosine: 0.637244
# token_id: 266, token: "╰", head: 6, cosine: 0.637949
# token_id: 367, token: "╴", head: 7, cosine: 0.638395
# token_id: 438, token: "┌", head: 7, cosine: 0.642260
# token_id: 420, token: "⁃", head: 0, cosine: 0.642357
# token_id: 202, token: "㇓", head: 9, cosine: 0.642853
# token_id: 266, token: "╰", head: 1, cosine: 0.644336
# token_id: 127, token: "ö", head: 4, cosine: 0.645651
# token_id: 148, token: "╾", head: 6, cosine: 0.645887
# token_id: 77, token: "┯", head: 5, cosine: 0.649377
# token_id: 83, token: "▼", head: 3, cosine: 0.649962
# token_id: 111, token: "├", head: 3, cosine: 0.651766
# token_id: 77, token: "┯", head: 4, cosine: 0.652354
# token_id: 77, token: "┯", head: 8, cosine: 0.653671
# token_id: 254, token: "€", head: 8, cosine: 0.653814
# token_id: 457, token: "Ü", head: 7, cosine: 0.656881
# token_id: 352, token: "∧", head: 2, cosine: 0.656967
# token_id: 199, token: "┤", head: 3, cosine: 0.657137
# token_id: 209, token: "┃", head: 11, cosine: 0.658226
# token_id: 100, token: "Ö", head: 4, cosine: 0.658887
# token_id: 199, token: "┤", head: 4, cosine: 0.660593
# token_id: 127, token: "ö", head: 0, cosine: 0.662292
# token_id: 341, token: "’", head: 4, cosine: 0.662532
# token_id: 341, token: "’", head: 11, cosine: 0.663570
# token_id: 77, token: "┯", head: 1, cosine: 0.663739
# token_id: 27, token: "·", head: 2, cosine: 0.665503
# token_id: 254, token: "€", head: 5, cosine: 0.665820
# token_id: 204, token: "∨", head: 11, cosine: 0.668643
# token_id: 148, token: "╾", head: 11, cosine: 0.668934
# token_id: 26, token: "┥", head: 0, cosine: 0.669182
# token_id: 148, token: "╾", head: 4, cosine: 0.670402
# token_id: 199, token: "┤", head: 5, cosine: 0.670908
# token_id: 339, token: "~", head: 5, cosine: 0.675899
# token_id: 148, token: "╾", head: 10, cosine: 0.676028
# token_id: 305, token: "µ", head: 5, cosine: 0.677594
# token_id: 202, token: "㇓", head: 6, cosine: 0.677703
# token_id: 338, token: "┗", head: 1, cosine: 0.680237
# token_id: 343, token: "🌸", head: 0, cosine: 0.680242
# token_id: 204, token: "∨", head: 1, cosine: 0.682378
# token_id: 51, token: "┛", head: 5, cosine: 0.683530
# token_id: 372, token: "┙", head: 8, cosine: 0.685659
# token_id: 266, token: "╰", head: 3, cosine: 0.690018
# token_id: 339, token: "~", head: 9, cosine: 0.692101
# token_id: 343, token: "🌸", head: 3, cosine: 0.692360
# token_id: 209, token: "┃", head: 9, cosine: 0.692681
# token_id: 27, token: "·", head: 6, cosine: 0.693860
# token_id: 341, token: "’", head: 1, cosine: 0.694489
# token_id: 266, token: "╰", head: 8, cosine: 0.695397

cargo run --release -- compare model-l9-3.dat model-l9-4.dat --limit 100;
# share layer_0..=layer_7
# key: norm_8_bias, cosine: -0.057946
# key: feedforward1_8_bias, cosine: -0.038794
# key: head_8_11_q, cosine: -0.038251
# key: head_8_3_k, cosine: -0.019286
# key: head_8_7_q, cosine: -0.016760
# key: head_8_2_q, cosine: -0.014562
# key: head_8_9_v, cosine: -0.012086
# key: head_8_0_q, cosine: -0.009669
# key: head_8_9_q, cosine: -0.009395
# key: head_8_6_q, cosine: -0.009222
# key: head_8_5_q, cosine: -0.007791
# key: head_8_10_v, cosine: -0.006408
# key: head_8_4_k, cosine: -0.005703
# key: head_8_1_k, cosine: -0.004858
# key: head_8_8_q, cosine: -0.004275
# key: head_8_7_v, cosine: -0.003158
# key: proj_8_weights, cosine: -0.001450
# key: head_8_11_v, cosine: 0.000311
# key: feedforward2_8_weights, cosine: 0.000852
# key: head_8_5_v, cosine: 0.002100
# key: head_8_1_q, cosine: 0.002113
# key: feedforward1_8_weights, cosine: 0.002668
# key: head_8_1_v, cosine: 0.003338
# key: head_8_4_q, cosine: 0.003506
# key: head_8_0_k, cosine: 0.008435
# key: head_8_11_k, cosine: 0.008571
# key: head_8_7_k, cosine: 0.009105
# key: head_8_9_k, cosine: 0.009960
# key: head_8_4_v, cosine: 0.010336
# key: proj_8_bias, cosine: 0.010535
# key: head_8_8_v, cosine: 0.011005
# key: head_8_0_v, cosine: 0.011371
# key: head_8_5_k, cosine: 0.012035
# key: head_8_2_k, cosine: 0.012384
# key: head_8_10_q, cosine: 0.013797
# key: head_8_6_v, cosine: 0.014542
# key: head_8_8_k, cosine: 0.015774
# key: head_8_3_v, cosine: 0.015836
# key: head_8_6_k, cosine: 0.016891
# key: head_8_2_v, cosine: 0.019569
# key: head_8_3_q, cosine: 0.022896
# key: head_8_10_k, cosine: 0.023344
# key: atten_norm_8_coeff, cosine: 0.042450
# key: norm_8_coeff, cosine: 0.100365
# key: atten_norm_8_bias, cosine: 0.378589
# key: atten_norm_7_bias, cosine: 0.511562
# key: feedforward2_8_bias, cosine: 0.527666
# key: proj_7_bias, cosine: 0.561215
# key: feedforward1_7_bias, cosine: 0.575634
# key: feedforward2_7_bias, cosine: 0.663957
# key: feedforward1_6_bias, cosine: 0.682216
# key: feedforward1_5_bias, cosine: 0.704578
# key: norm_7_bias, cosine: 0.721659
# key: atten_norm_6_bias, cosine: 0.808264
# key: head_7_1_q, cosine: 0.817561
# key: head_7_1_k, cosine: 0.859538
# key: head_7_3_k, cosine: 0.865227
# key: feedforward2_5_bias, cosine: 0.866691
# key: atten_norm_5_bias, cosine: 0.883100
# key: head_7_3_q, cosine: 0.884407
# key: feedforward2_6_bias, cosine: 0.886891
# key: proj_6_bias, cosine: 0.889976
# key: proj_5_bias, cosine: 0.891322
# key: head_7_6_q, cosine: 0.893134
# key: head_7_11_q, cosine: 0.901185
# key: head_7_2_k, cosine: 0.907852
# key: head_7_0_k, cosine: 0.907874
# key: head_7_6_k, cosine: 0.911410
# key: norm_6_bias, cosine: 0.911560
# key: head_7_0_q, cosine: 0.911980
# key: head_7_11_k, cosine: 0.914451
# key: head_7_2_q, cosine: 0.925968
# key: head_7_5_k, cosine: 0.927130
# key: head_7_4_q, cosine: 0.927388
# key: head_7_4_k, cosine: 0.928629
# key: head_7_5_q, cosine: 0.933021
# key: head_7_7_q, cosine: 0.934308
# key: feedforward2_7_weights, cosine: 0.934609
# key: norm_5_bias, cosine: 0.936133
# key: head_7_9_q, cosine: 0.936570
# key: head_7_7_k, cosine: 0.937137
# key: norm_7_coeff, cosine: 0.937675
# key: head_7_8_q, cosine: 0.938015
# key: proj_7_weights, cosine: 0.938509
# key: proj_4_bias, cosine: 0.938990
# key: head_7_9_k, cosine: 0.939521
# key: head_6_5_q, cosine: 0.941954
# key: head_7_8_k, cosine: 0.943340
# key: head_6_4_q, cosine: 0.944930
# key: head_6_9_q, cosine: 0.947263
# key: atten_norm_7_coeff, cosine: 0.947629
# key: head_7_3_v, cosine: 0.947937
# key: head_6_6_q, cosine: 0.948830
# key: head_7_7_v, cosine: 0.949129
# key: head_6_7_k, cosine: 0.949439
# key: feedforward1_4_bias, cosine: 0.949905
# key: head_6_3_q, cosine: 0.950423
# key: head_7_1_v, cosine: 0.950918
# key: head_6_6_k, cosine: 0.951625
# key: head_7_0_v, cosine: 0.952006
# token_id: 180, token: "✕", head: 4, cosine: 0.573949
# token_id: 180, token: "✕", head: 0, cosine: 0.620374
# token_id: 180, token: "✕", head: 1, cosine: 0.652082
# token_id: 114, token: "┬", head: 11, cosine: 0.657293
# token_id: 177, token: "❨", head: 7, cosine: 0.694243
# token_id: 469, token: "［", head: 10, cosine: 0.696849
# token_id: 129, token: "﴿", head: 7, cosine: 0.697278
# token_id: 416, token: "►", head: 2, cosine: 0.697551
# token_id: 267, token: "ï", head: 10, cosine: 0.697714
# token_id: 438, token: "┌", head: 9, cosine: 0.701539
# token_id: 491, token: "©", head: 0, cosine: 0.709762
# token_id: 491, token: "©", head: 5, cosine: 0.713769
# token_id: 237, token: "┘", head: 7, cosine: 0.715963
# token_id: 29, token: "─", head: 4, cosine: 0.719807
# token_id: 292, token: "〙", head: 4, cosine: 0.719841
# token_id: 390, token: "〗", head: 6, cosine: 0.726122
# token_id: 180, token: "✕", head: 8, cosine: 0.727053
# token_id: 438, token: "┌", head: 4, cosine: 0.728946
# token_id: 267, token: "ï", head: 2, cosine: 0.729097
# token_id: 367, token: "╴", head: 9, cosine: 0.731825
# token_id: 438, token: "┌", head: 3, cosine: 0.732434
# token_id: 213, token: "❩", head: 9, cosine: 0.735256
# token_id: 114, token: "┬", head: 5, cosine: 0.736556
# token_id: 303, token: "%", head: 7, cosine: 0.737915
# token_id: 180, token: "✕", head: 9, cosine: 0.739470
# token_id: 303, token: "%", head: 4, cosine: 0.742138
# token_id: 267, token: "ï", head: 3, cosine: 0.745122
# token_id: 129, token: "﴿", head: 8, cosine: 0.749237
# token_id: 491, token: "©", head: 8, cosine: 0.751173
# token_id: 114, token: "┬", head: 9, cosine: 0.751754
# token_id: 292, token: "〙", head: 3, cosine: 0.757642
# token_id: 180, token: "✕", head: 6, cosine: 0.758703
# token_id: 114, token: "┬", head: 4, cosine: 0.760167
# token_id: 159, token: "™", head: 3, cosine: 0.763138
# token_id: 180, token: "✕", head: 10, cosine: 0.763679
# token_id: 267, token: "ï", head: 1, cosine: 0.764556
# token_id: 109, token: "｝", head: 0, cosine: 0.764886
# token_id: 159, token: "™", head: 11, cosine: 0.766093
# token_id: 237, token: "┘", head: 4, cosine: 0.766608
# token_id: 469, token: "［", head: 1, cosine: 0.769821
# token_id: 466, token: "〛", head: 1, cosine: 0.772947
# token_id: 111, token: "├", head: 3, cosine: 0.773393
# token_id: 158, token: "〕", head: 8, cosine: 0.773602
# token_id: 180, token: "✕", head: 3, cosine: 0.774643
# token_id: 416, token: "►", head: 10, cosine: 0.775230
# token_id: 303, token: "%", head: 10, cosine: 0.775873
# token_id: 296, token: "᾽", head: 8, cosine: 0.776136
# token_id: 213, token: "❩", head: 11, cosine: 0.776810
# token_id: 158, token: "〕", head: 7, cosine: 0.776842
# token_id: 416, token: "►", head: 4, cosine: 0.777834
# token_id: 266, token: "╰", head: 2, cosine: 0.779191
# token_id: 267, token: "ï", head: 5, cosine: 0.779400
# token_id: 438, token: "┌", head: 7, cosine: 0.780140
# token_id: 303, token: "%", head: 9, cosine: 0.780660
# token_id: 31, token: "𝄔", head: 5, cosine: 0.781070
# token_id: 491, token: "©", head: 3, cosine: 0.783032
# token_id: 206, token: "X", head: 3, cosine: 0.784441
# token_id: 180, token: "✕", head: 11, cosine: 0.784720
# token_id: 114, token: "┬", head: 2, cosine: 0.786034
# token_id: 252, token: "΄", head: 1, cosine: 0.786475
# token_id: 491, token: "©", head: 10, cosine: 0.787121
# token_id: 466, token: "〛", head: 5, cosine: 0.787966
# token_id: 394, token: "┐", head: 6, cosine: 0.788594
# token_id: 438, token: "┌", head: 10, cosine: 0.790984
# token_id: 438, token: "┌", head: 0, cosine: 0.792872
# token_id: 496, token: "▸", head: 1, cosine: 0.793413
# token_id: 158, token: "〕", head: 3, cosine: 0.793557
# token_id: 114, token: "┬", head: 8, cosine: 0.795889
# token_id: 25, token: "）", head: 10, cosine: 0.796495
# token_id: 109, token: "｝", head: 8, cosine: 0.796980
# token_id: 436, token: "╭", head: 1, cosine: 0.797041
# token_id: 177, token: "❨", head: 1, cosine: 0.797907
# token_id: 173, token: "❲", head: 10, cosine: 0.798451
# token_id: 213, token: "❩", head: 1, cosine: 0.799034
# token_id: 8, token: "7", head: 11, cosine: 0.799272
# token_id: 199, token: "┤", head: 4, cosine: 0.801125
# token_id: 469, token: "［", head: 0, cosine: 0.801775
# token_id: 158, token: "〕", head: 0, cosine: 0.803384
# token_id: 237, token: "┘", head: 1, cosine: 0.803650
# token_id: 390, token: "〗", head: 1, cosine: 0.804509
# token_id: 238, token: "及", head: 4, cosine: 0.805136
# token_id: 114, token: "┬", head: 7, cosine: 0.805343
# token_id: 114, token: "┬", head: 6, cosine: 0.805347
# token_id: 363, token: "H", head: 9, cosine: 0.806237
# token_id: 180, token: "✕", head: 5, cosine: 0.806287
# token_id: 438, token: "┌", head: 5, cosine: 0.807388
# token_id: 180, token: "✕", head: 2, cosine: 0.808912
# token_id: 109, token: "｝", head: 9, cosine: 0.809158
# token_id: 416, token: "►", head: 6, cosine: 0.809408
# token_id: 294, token: "Z", head: 7, cosine: 0.809942
# token_id: 416, token: "►", head: 9, cosine: 0.811265
# token_id: 238, token: "及", head: 8, cosine: 0.811627
# token_id: 252, token: "΄", head: 10, cosine: 0.812064
# token_id: 159, token: "™", head: 2, cosine: 0.812270
# token_id: 255, token: "⊥", head: 5, cosine: 0.812519
# token_id: 237, token: "┘", head: 10, cosine: 0.813294
# token_id: 255, token: "⊥", head: 2, cosine: 0.814076
# token_id: 267, token: "ï", head: 0, cosine: 0.814511
# token_id: 213, token: "❩", head: 7, cosine: 0.814816
# token_id: 159, token: "™", head: 8, cosine: 0.816285

cp model-l9-1.dat model-l9.dat;

cargo run --release -- insert-layer --input model-l9.dat --output model-l10-1.dat --insert-at 9;
cargo run --release -- insert-layer --input model-l9.dat --output model-l10-2.dat --insert-at 9;

for _ in 0..8 {
    cargo run --release -- train --model model-l10-1.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l10-2.dat --steps 31;
    sleep 200sec;
}

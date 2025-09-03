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

for _ in 0..8 {
    cargo run --release -- train --model model-l9-1.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-2.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-3.dat --steps 31;
    sleep 200sec;
    cargo run --release -- train --model model-l9-4.dat --steps 31;
    sleep 200sec;
}

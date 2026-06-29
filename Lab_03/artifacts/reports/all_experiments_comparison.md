# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | classical_base | ninguno | COMPLETADO | 0.8698 | 0.8698 | 10.3337 | 13.4507 | 0.5245 | - | 59.8807 |  |
| E1 | classical_pca_low | PCA=50 componentes (informacion insuficiente) | COMPLETADO | 0.8257 | 0.8256 | 11.9093 | 15.3568 | 0.3802 | - | 32.8303 |  |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8313 | 0.8314 | 17.5779 | 23.6237 | -0.4666 | 77202947 | 0.0000 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8572 | 0.8572 | 11.8745 | 16.0838 | 0.3202 | 77202947 | 0.0000 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8288 | 0.8287 | 20.3057 | 27.3807 | -0.9702 | 77202947 | 0.0000 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8310 | 0.8312 | 12.5399 | 17.5826 | 0.1876 | 77202947 | 0.0000 |  |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 0.0000 |  |
| E3 | cnn_no_augmentation | sin aumentacion | COMPLETADO | 0.8766 | 0.8762 | 9.6720 | 13.5238 | 0.5194 | 286755 | 0.0000 |  |
| E3 | cnn_no_dropout | dropout=0.0 | COMPLETADO | 0.8811 | 0.8808 | 9.4201 | 12.9366 | 0.5602 | 286755 | 0.0000 |  |
| E3 | cnn_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8760 | 0.8757 | 12.8291 | 18.3405 | 0.1160 | 286755 | 0.0000 |  |
| E3 | cnn_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8521 | 0.8519 | 9.3570 | 12.9573 | 0.5588 | 286755 | 0.0000 |  |
| E3 | cnn_age_normalized | edad normalizada a [0,1], lambda=1.0 | COMPLETADO | 0.8684 | 0.8685 | 10.2140 | 13.6030 | 0.5137 | 286755 | 0.0000 |  |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8212 | 0.8209 | 10.9718 | 14.6851 | 0.4333 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8234 | 0.8235 | 10.9784 | 14.6321 | 0.4373 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8212 | 0.8209 | 11.5166 | 15.5129 | 0.3676 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8212 | 0.8209 | 10.9168 | 14.5914 | 0.4405 | 1539 | 0.0000 |  |
| E5 | resnet_finetuning_base | layer4 descongelado | COMPLETADO | 0.9193 | 0.9194 | 6.4568 | 9.1025 | 0.7823 | 8395267 | 0.0000 |  |
| E5 | resnet_finetuning_unfreeze_more | layer3+layer4 descongelados | COMPLETADO | 0.9221 | 0.9222 | 6.4029 | 9.1259 | 0.7811 | 10494979 | 0.0000 |  |
| E5 | resnet_finetuning_lr_low | learning rate 1e-4 | COMPLETADO | 0.9131 | 0.9132 | 7.2907 | 9.8579 | 0.7446 | 8395267 | 0.0000 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9185 | 0.9184 | 5.6027 | 8.0650 | 0.8291 | 8395267 | 0.0000 |  |
| E6 | cnn_aligned_base | datos raw alineados automaticamente, sin aumentacion | COMPLETADO | 0.7919 | 0.7920 | 12.8777 | 17.5472 | 0.2572 | 286755 | 0.0000 |  |
| E6 | resnet_aligned_base | ResNet18 fine-tuning sobre raw+FaceAligner | COMPLETADO | 0.8862 | 0.8861 | 8.3888 | 12.3616 | 0.6313 | 8395267 | 2298.6979 |  |

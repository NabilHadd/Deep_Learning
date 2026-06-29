# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | classical_base | ninguno | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E1 | classical_pca_low | PCA=50 componentes (informacion insuficiente) | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E2 | mlp_base | ninguno | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E2 | mlp_no_dropout | dropout=0.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E2 | mlp_lambda_low | lambda_age=0.001 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E2 | mlp_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 135.8467 |  |
| E3 | cnn_no_augmentation | sin aumentacion | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_no_dropout | dropout=0.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_low | lambda_age=0.001 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_age_normalized | edad normalizada a [0,1], lambda=1.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E4 | resnet_frozen_base | ninguno | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E5 | resnet_finetuning_base | layer4 descongelado | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E5 | resnet_finetuning_unfreeze_more | layer3+layer4 descongelados | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E5 | resnet_finetuning_lr_low | learning rate 1e-4 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E6 | cnn_aligned_base | datos raw alineados automaticamente, sin aumentacion | COMPLETADO | 0.7923 | 0.7923 | 12.8711 | 17.5390 | 0.2579 | 286755 | 2318.2970 |  |

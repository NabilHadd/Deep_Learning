# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E5 | resnet_finetuning_base | layer4 descongelado | COMPLETADO | 0.9193 | 0.9194 | 6.4568 | 9.1025 | 0.7823 | 8395267 | 0.0000 |  |
| E5 | resnet_finetuning_unfreeze_more | layer3+layer4 descongelados | COMPLETADO | 0.9221 | 0.9222 | 6.4029 | 9.1259 | 0.7811 | 10494979 | 0.0000 |  |
| E5 | resnet_finetuning_lr_low | learning rate 1e-4 | COMPLETADO | 0.9131 | 0.9132 | 7.2907 | 9.8579 | 0.7446 | 8395267 | 0.0000 |  |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | COMPLETADO | 0.9185 | 0.9184 | 5.6027 | 8.0650 | 0.8291 | 8395267 | 0.0000 |  |

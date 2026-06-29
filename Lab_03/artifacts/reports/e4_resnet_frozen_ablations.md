# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E4 | resnet_frozen_base | ninguno | COMPLETADO | 0.8212 | 0.8209 | 10.9718 | 14.6851 | 0.4333 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | COMPLETADO | 0.8234 | 0.8235 | 10.9784 | 14.6321 | 0.4373 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8212 | 0.8209 | 11.5166 | 15.5129 | 0.3676 | 1539 | 0.0000 |  |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8212 | 0.8209 | 10.9168 | 14.5914 | 0.4405 | 1539 | 0.0000 |  |

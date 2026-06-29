# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 | classical_base | ninguno | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E1 | classical_pca_50 | PCA=50 componentes | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E1 | classical_pca_200 | PCA=200 componentes | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E2 | mlp_base | ninguno | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E2 | mlp_no_dropout | dropout=0.0 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E2 | mlp_lambda_low | lambda_age=0.001 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E2 | mlp_lambda_high | lambda_age=0.1 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8746 | 9.4320 | 13.2124 | 0.5412 | 286755 | 529.5622 |  |
| E3 | cnn_no_augmentation | sin aumentacion | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_no_dropout | dropout=0.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_low | lambda_age=0.001 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E4 | resnet_frozen_base | ninguno | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E4 | resnet_frozen_no_augmentation | sin aumentacion | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E4 | resnet_frozen_lambda_low | lambda_age=0.001 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E4 | resnet_frozen_lambda_high | lambda_age=0.1 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E5 | resnet_finetuning_base | ninguno | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E5 | resnet_finetuning_unfreeze_more | mas bloques descongelados | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E5 | resnet_finetuning_lr_low | learning rate menor | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |
| E5 | resnet_finetuning_lambda_high | lambda_age=0.1 | NO_IMPLEMENTADO | - | - | - | - | - | - | - | El experimento debe ser completado por los alumnos. |

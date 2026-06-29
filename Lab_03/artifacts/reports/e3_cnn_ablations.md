# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E3 | cnn_base | ninguno | COMPLETADO | 0.8746 | 0.8744 | 9.6607 | 13.2543 | 0.5383 | 286755 | 135.8467 |  |
| E3 | cnn_no_augmentation | sin aumentacion | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_no_dropout | dropout=0.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_low | lambda_age=0.001 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_lambda_high | lambda_age=0.1 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |
| E3 | cnn_age_normalized | edad normalizada a [0,1], lambda=1.0 | NO_EJECUTADO | - | - | - | - | - | - | - | Implementado, pero no fue seleccionado en esta ejecucion. |

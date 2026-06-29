# Reporte de experimentos

| strategy_id | experiment_name | changed_component | status | gender_accuracy | gender_f1 | age_mae | age_rmse | age_r2 | trainable_parameters | training_seconds | message |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E2 | mlp_base | ninguno | COMPLETADO | 0.8313 | 0.8314 | 17.5779 | 23.6237 | -0.4666 | 77202947 | 0.0000 |  |
| E2 | mlp_no_dropout | dropout=0.0 | COMPLETADO | 0.8572 | 0.8572 | 11.8745 | 16.0838 | 0.3202 | 77202947 | 0.0000 |  |
| E2 | mlp_lambda_low | lambda_age=0.001 | COMPLETADO | 0.8288 | 0.8287 | 20.3057 | 27.3807 | -0.9702 | 77202947 | 0.0000 |  |
| E2 | mlp_lambda_high | lambda_age=0.1 | COMPLETADO | 0.8310 | 0.8312 | 12.5399 | 17.5826 | 0.1876 | 77202947 | 0.0000 |  |

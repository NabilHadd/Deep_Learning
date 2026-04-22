from sklearn.model_selection import train_test_split

from data_loader import data_loader
from visualization import frec_plot, entropy_plot, plot_roc_multiclass, confusion_matrix_plot
import eda
import train as tr
from stacking import stacking
from bagging import bagging
from boosting import boosting
from config import DATA_PATH, GDS_R2_RESULTS_PATH, GDS_RESULTS_PATH, RAW_DATA_PATH,  GDS_CLASSES, GDS_R2_CLASSES, SAVE_IMAGES_PATH, Model_names, MODELS






#Leemos los datos.
df_15_raw = data_loader(RAW_DATA_PATH)

#identificamos outliers y missing values:
eda.ouliers_seeker(df_15_raw, 0, df_15_raw.columns.to_list().index('GDS'))
eda.missing_values(df_15_raw)
eda.feauters_labels_count(df_15_raw)



#aplicamos un procesamiento general en función del analisis necesario solo la primera vez
# eda.make_csv(df_15_raw, DATA_PATH) 

#Leemos los datos procesados
df_15       =       data_loader(DATA_PATH)

#consideramos solo los labels haciendo un split desde el indice de gds en adelante
labels      =       df_15.columns.tolist()[df_15.columns.to_list().index('GDS'):]

X           =       df_15.drop(columns=labels) #subdataframe con los atributos
Y_GDS       =       df_15[['GDS']].values #subdataframe con para gds
Y_GDS_R2    =       df_15[['GDS_R2']].values #subdataframe con para gds_r2

X_TRAIN, X_TEST, Y_TRAIN_GDS, Y_TEST_GDS = train_test_split(X, Y_GDS, test_size=0.4, random_state=0, stratify=Y_GDS)
X_TRAIN, X_TEST, Y_TRAIN_GDS_R2, Y_TEST_GDS_R2 = train_test_split(X, Y_GDS_R2, test_size=0.4, random_state=0, stratify=Y_GDS_R2)


frec_plot(df_15)
entropy_plot(df_15[labels])
#debido a que la entropia define desorden, ejemplo (el desorden perfecto para 2 clases son frecuencias de 50/50)
#En este contexto la entropia representa balanceo, por lo que a mayor entropria mas balanceadas estan las clases.


#PARA GDS
results_dataframe = tr.random_train(X_TRAIN, Y_TRAIN_GDS, X_TEST, Y_TEST_GDS)
eda.make_csv(results_dataframe[0], GDS_RESULTS_PATH, idx=True)

#PARA GDS_R2
results_dataframe = tr.random_train(X_TRAIN, Y_TRAIN_GDS_R2, X_TEST, Y_TEST_GDS_R2)
eda.make_csv(results_dataframe[0], GDS_R2_RESULTS_PATH, idx=True)


#Debido a que ya creamos los modelos, ya no es necesario correr el random search. ahora extraemos los datos desde los dataframes guardados.
gds_ensamble_models = {model_name: MODELS[model_name](GDS_RESULTS_PATH, X_train=X_TRAIN, Y_train=Y_TRAIN_GDS, X_test=X_TEST, Y_test=Y_TEST_GDS)
                        for model_name in Model_names}
gds_r2_ensamble_models = {model_name: MODELS[model_name](GDS_R2_RESULTS_PATH, X_train=X_TRAIN, Y_train=Y_TRAIN_GDS_R2, X_test=X_TEST, Y_test=Y_TEST_GDS_R2)
                            for model_name in Model_names}


for name, model in gds_ensamble_models.items():
    plot_roc_multiclass(
        model=model, 
        model_name=name.value, 
        X_test=X_TEST, y_test=Y_TEST_GDS, 
        classes_names=GDS_CLASSES, 
        save_path=SAVE_IMAGES_PATH / "GDS" / f"{name.value}_roc_gds.png"
        )

for name, model in gds_r2_ensamble_models.items():
    plot_roc_multiclass(model=model, 
                        model_name=name.value, 
                        X_test=X_TEST, 
                        y_test=Y_TEST_GDS_R2, 
                        classes_names=GDS_R2_CLASSES, 
                        save_path=SAVE_IMAGES_PATH / "GDS_R2" / f"{name.value}_roc_gds_r2.png"
                        )

    confusion_matrix_plot(model=model, 
                            model_name=name.value, 
                            X_test=X_TEST,
                            y_test=Y_TEST_GDS_R2,
                            classes_names=GDS_R2_CLASSES,
                            save_path=SAVE_IMAGES_PATH / "GDS_R2" / f"{name.value}_cm_gds_r2.png"
                            )

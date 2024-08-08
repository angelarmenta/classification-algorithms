#developed by Roberto Ángel Meléndez-Armenta
#https://www.youtube.com/@educar-ia

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos
file_path = '/Users/angelarmenta/Dropbox/educar-ia/Clasificacion/Student_performance_data _.csv'
data = pd.read_csv(file_path)

# Mostrar la información general del dataset
print(data.info())

# Mostrar los primeros 5 registros
print(data.head())

# Verificar valores nulos
print(data.isnull().sum())

# Separar características y variable objetivo
X = data.drop(columns=['StudentID', 'GradeClass'])
y = data['GradeClass']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Graficar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Actual')
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Carregamento do dataset (substitua 'seu_arquivo.csv' pelo nome do seu arquivo)
data = pd.read_csv('card_transdata.csv')

# Seleção das features (substitua com as colunas apropriadas do seu dataset)
X = data[['distance_from_home', 'used_pin_number']]
y = data['fraud']

# Divisão dos dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronização das features (é importante para o KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinamento do Modelo Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)

# Avaliação do Modelo Naive Bayes
print("Resultado do Naive Bayes:")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Classification Report:\n", classification_report(y_test, nb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))

# Treinamento do Modelo Árvore de Decisão
tree_classifier = DecisionTreeClassifier(max_depth=5)
tree_classifier.fit(X_train, y_train)
tree_predictions = tree_classifier.predict(X_test)

# Avaliação do Modelo Árvore de Decisão
print("Resultado da Árvore de Decisão:")
print("Accuracy:", accuracy_score(y_test, tree_predictions))
print("Classification Report:\n", classification_report(y_test, tree_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, tree_predictions))

# Treinamento do Modelo KNN
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)

# Avaliação do Modelo KNN
print("Resultado do KNN:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Classification Report:\n", classification_report(y_test, knn_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))

# Função para plotar as fronteiras de decisão
def plot_decision_boundary(X, y, classifier, title):
    h = .6  # Tamanho do passo na malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    plt.title(title)

# Plotando as fronteiras de decisão usando as duas primeiras features
X_train_plot = X_train[:, :2]
X_test_plot = X_test[:, :2]

plt.figure(figsize=(15, 5))

# Fronteira de decisão do Naive Bayes
plt.subplot(131)
plot_decision_boundary(X_train_plot, y_train, nb_classifier, 'Fronteira de Decisão - Naive Bayes')

# Fronteira de decisão da Árvore de Decisão
plt.subplot(132)
plot_decision_boundary(X_train_plot, y_train, tree_classifier, 'Fronteira de Decisão - Árvore de Decisão')

# Fronteira de decisão do KNN
plt.subplot(133)
plot_decision_boundary(X_train_plot, y_train, knn_classifier, 'Fronteira de Decisão - KNN')

plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xlsxwriter.workbook import Workbook
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import itertools

class FeatureExtractionUnsupervised:
    def __init__(self, data_path, target_column_index=36, n_components=10):
        self.data_path = data_path
        self.target_column_index = target_column_index
        self.n_components = n_components

    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        self.X = self.data.drop(columns=[self.data.columns[self.target_column_index]])
        self.y = self.data[self.data.columns[self.target_column_index]]

    def feature_extraction(self):
        # Standardize the data
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

        # Perform PCA for feature extraction
        pca = PCA(n_components=self.n_components)
        self.X_pca = pca.fit_transform(self.X_scaled)

        # Create a DataFrame with the principal components
        self.pca_components_df = pd.DataFrame(data=self.X_pca, columns=[f"PC{i+1}" for i in range(self.n_components)])

    def save_to_excel(self, output_path):
        # Create a new DataFrame with principal components and the target column
        pca_data = pd.concat([self.pca_components_df, self.y], axis=1)

        # Write to Excel
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            pca_data.to_excel(writer, sheet_name='PCA_Features', index=False)

        print("PCA features saved to Excel.")



class FeatureClassificationDeepLearning:
    def __init__(self, data_path, target_column_index=10, n_features=10, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target_column_index = target_column_index
        self.n_features = n_features
        self.test_size = test_size
        self.random_state = random_state

   
    def load_data(self):
        data = pd.read_excel(self.data_path)
        self.X = data.drop(columns=[data.columns[self.target_column_index]])
        self.y = data[data.columns[self.target_column_index]]
        self.y_encode = LabelEncoder()
        self.y_encoded = self.y_encode.fit_transform(self.y)
        print(self.y_encode.classes_)


    def train_model(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y_encoded, test_size=self.test_size, random_state=self.random_state
        )
        # Build the neural network model
        model = Sequential()
        model.add(Dense(64, input_dim=self.n_features, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.y_encode.classes_), activation='softmax'))

        # Compile the model
        model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_split=0.1)

        # Evaluate the model on the test set
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test accuracy: {accuracy}")
        model.summary()
        from keras.utils.vis_utils import plot_model
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
        cm = confusion_matrix(y_test, y_pred_classes)

        # Get the class labels (assuming the classes are named 'Healthy', '1_BB', '2_BB', '3_BB')
        class_labels = ['Healthy', '1_BB', '2_BB', '3_BB']

        # Plot the confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=class_labels, yticklabels=class_labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig('matrix.png',dpi=300)
        plt.show()
        self.plot_confusion_matrix(cm,class_labels,filename=f'new_cm.png')
    
    def plot_confusion_matrix(self,cm, class_names, filename):
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Compute the labels from the normalized confusion matrix.
        labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        plt.savefig(filename)
        return figure

def Visualize(data):

    data = data
    data = pd.read_csv(data)

    # Step 2: Extract the data from columns 1 and 36
    column1_index = 34
    column36_index = 36

    # Step 3: Extract the data from columns 1 and 36
    column1_data = data[data.columns[column1_index]]
    column36_data = data[data.columns[column36_index]]

    # Step 4: Create a box plot to visualize the relationship between Column 1 and Column 36
    plt.figure(figsize=(8, 6))
    plt.boxplot([column1_data[data[data.columns[column36_index]] == category] for category in column36_data.unique()], labels=column36_data.unique())
    plt.xlabel('Speed')
    plt.ylabel('Class')
    plt.title('Relationship between Rotor Speed and Broken Bars')
    plt.grid(True)
    plt.savefig('speed.png',dpi=300)

def Visually(data):
    data = pd.read_csv(data_path, header=None)

    # Step 2: Define the column indices and the target column index
    feature_indices = [1, 2,3]
    target_index = 36

    # Step 3: Extract the data for features and target
    features = data[data.columns[feature_indices]]
    target = data[data.columns[target_index]]

    # Step 4: Combine features and target into a single DataFrame for visualization
    data_for_viz = pd.concat([features, target], axis=1)

    # Step 5: Create a pair plot
    for feature_idx in feature_indices:
        plt.figure(figsize=(8, 6))
        sns.violinplot(x=data.columns[target_index], y=data.columns[feature_idx], data=data_for_viz)
        plt.xlabel('Classification')
        plt.ylabel(f'Feature {feature_idx + 1}')
        plt.title(f'Feature {feature_idx + 1} vs. Classification')
        plt.show()



# Example usage:

data_path = "C:\\Users\\Edwin\\Downloads\\Broken_bars_data.csv"
output_path_pca = "C:\\Users\\Edwin\\Downloads\\pca_features.xlsx"
'''
unsupervised_fe = FeatureExtractionUnsupervised(data_path)
unsupervised_fe.load_data()
unsupervised_fe.feature_extraction()
unsupervised_fe.save_to_excel(output_path_pca)
'''
data_path = output_path_pca

deep_learning_classifier = FeatureClassificationDeepLearning(data_path)
deep_learning_classifier.load_data()
deep_learning_classifier.train_model()


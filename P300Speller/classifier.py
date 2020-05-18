# Importing the libraries
import os
# import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.preprocessing import StandardScaler
from tkinter import messagebox
from sklearn.model_selection import GridSearchCV
import pickle
from keras.models import Sequential
from keras.layers import Dense

clf = []
sc = None
filename = None
saveModel = False


def wrapper(case, sampler=1, path="", clf_type=1):
    global clf, names, saveModel
    clf = []
    print("Case: {} Sampler: {}".format(case, sampler))
    ##clf_type = 0 for using saved classifiers
    if (clf_type != 0):
        saveModel = messagebox.askyesno("Success", "Would you like to save the model?")
    filepath = os.path.join(os.getcwd(), 'Data')
    df = {}
    X_training = {}
    X_train = {}
    y_train = {}
    i = 0

    if (clf_type == 0):

        if (case == 1):
            fpath = os.path.join('./Classifier/Combined', "combined_classifier.sav")
            clf.append(pickle.load(open(fpath, 'rb')))
            print("File name : {} is loaded".format("combined_classifier.sav"))

        elif (case == 2):
            loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Session'))
            for folder in os.listdir(loc):
                files = os.path.join(loc, folder)
                for file in os.listdir(files):
                    fpath = os.path.join(files, file)
                    clf.append(pickle.load(open(fpath, 'rb')))
                    print("File name : {} is loaded".format(file))

        elif (case == 3):
            loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Subject'))
            for file in os.listdir(loc):
                fpath = os.path.join(loc, file)
                clf.append(pickle.load(open(fpath, 'rb')))
                print("File name : {} is loaded".format(file))


    elif (case == 1):  # all data together

        for folder in os.listdir(filepath):

            files = os.path.join(filepath, folder)
            for file in os.listdir(files):
                # print("Filename is {}".format(file))
                dest = os.path.join(files, file)
                df[i] = pd.read_csv(dest, names=['Channel#1', 'Channel#2', 'Channel#3', 'Channel#4',
                                                 'Channel#5', 'Channel#6', 'Channel#7', 'Channel#8', 'Marker',
                                                 'Timestamp', 'ASCII', 'Epoch', 'Target'
                                                 ])
                i += 1

        data = df[0]
        for j in range(1, i):
            data = pd.concat([data, df[j]])

        X_training = data.iloc[:, 0:9].values
        y_train = data.iloc[:, 12].values

        # Feature Scaling
        sc = StandardScaler()

        X_train = X_training[:, 0:8]
        X_train = sc.fit_transform(X_train)
        trained_model = classifier(X_train, y_train, sampler, clf_type)
        if saveModel:
            filpath = os.path.join('./Classifier/Combined', "combined_classifier.sav")
            pickle.dump(trained_model, open(filpath, 'wb'))
        clf.append(trained_model)

    elif (case == 2):  # one session at a time
        for folder in os.listdir(filepath):
            files = os.path.join(filepath, folder)
            fpath = ""
            if saveModel:
                loc = os.path.join(os.getcwd(), os.path.join('Classifier', 'Session'))
                fpath = os.path.join(loc, folder)
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
            for file in os.listdir(files):
                dest = os.path.join(files, file)
                df[i] = pd.read_csv(dest, names=['Channel#1', 'Channel#2', 'Channel#3', 'Channel#4',
                                                 'Channel#5', 'Channel#6', 'Channel#7', 'Channel#8',
                                                 'Marker', 'Timestamp', 'ASCII', 'Epoch', 'Target'
                                                 ])

                X_training[i] = df[i].iloc[:, 0:9].values
                y_train[i] = df[i].iloc[:, 12].values
                print("File : {}".format(file))
                # Feature Scaling

                sc = StandardScaler()

                X_train[i] = X_training[i][:, 0:8]
                X_train[i] = sc.fit_transform(X_train[i])
                trained_model = classifier(X_train[i], y_train[i], sampler, clf_type)
                if saveModel:
                    filpath = os.path.join(fpath, os.path.splitext(file)[0] + ".sav")
                    pickle.dump(trained_model, open(filpath, 'wb'))
                clf.append(trained_model)
                i += 1

    elif (case == 3):  # each person at a time
        for folder in os.listdir(filepath):
            df = {}
            j = 0
            files = os.path.join(filepath, folder)
            fpath = ""
            if saveModel:
                fpath = os.path.join(os.getcwd(), os.path.join('Classifier', 'Subject'))
                if not os.path.exists(fpath):
                    os.makedirs(fpath)
            for file in os.listdir(files):
                dest = os.path.join(files, file)
                df[j] = pd.read_csv(dest, names=['Channel#1', 'Channel#2', 'Channel#3', 'Channel#4',
                                                 'Channel#5', 'Channel#6', 'Channel#7', 'Channel#8',
                                                 'Marker', 'Timestamp', 'ASCII', 'Epoch', 'Target'
                                                 ])
                j += 1
            data = df[0]

            for k in range(1, j):
                data = pd.concat([data, df[k]])

            X_training[i] = data.iloc[:, 0:9].values
            y_train[i] = data.iloc[:, 12].values

            # Feature Scaling

            sc = StandardScaler()

            X_train[i] = X_training[i][:, 0:8]
            X_train[i] = sc.fit_transform(X_train[i])
            trained_model = classifier(X_train[i], y_train[i], sampler, clf_type)
            if saveModel:
                filpath = os.path.join(fpath, folder + ".sav")
                pickle.dump(trained_model, open(filpath, 'wb'))
            clf.append(trained_model)
            i += 1
    elif (case == 4):
        df = pd.read_csv(path, names=['Channel#1', 'Channel#2', 'Channel#3', 'Channel#4', 'Channel#5', 'Channel#6',
                                      'Channel#7', 'Channel#8', 'Marker', 'Timestamp', 'ASCII', 'Epoch', 'Target'])

        X_train = df.iloc[:int(0.75 * df.shape[0]), 0:9].values
        y_train = df.iloc[:int(0.75 * df.shape[0]), 12].values
        X_test = df.iloc[int(0.75 * df.shape[0]):, 0:9].values
        y_test = df.iloc[int(0.75 * df.shape[0]):, 12].values

        sc = StandardScaler()

        X_train = sc.fit_transform(X_train)
        clf.append(classifier(X_train[:, 0:8], y_train, 1, clf_type))
        predictor(X_test, clf[0], y_test)

    messagebox.showinfo("Success", "Classifier is trained !!!")

    return clf


def classifier(X_train, y_train, sampler, clf_type=1):
    if (sampler == 0):
        X_resampled, y_resampled = X_train, y_train
    elif (sampler == 1):
        from imblearn.combine import SMOTEENN
        smote_enn = SMOTEENN(random_state=0)
        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    elif (sampler == 2):
        from imblearn.combine import SMOTETomek
        smote_tomek = SMOTETomek(random_state=0)
        X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
    elif (sampler == 3):
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=0)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
    elif (sampler == 4):
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    elif (sampler == 5):
        from imblearn.over_sampling import ADASYN
        X_resampled, y_resampled = ADASYN().fit_resample(X_train, y_train)

    if clf_type == 1:
        # Initialising the ANN
        classifier = Sequential()

        # Adding the input layer and the first hidden layer
        classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=8))

        # Adding the second hidden layer
        classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
        classifier.add(Dense(output_dim=4, init='uniform', activation='relu'))
        # classifier.add(Dense(output_dim = 2, init = 'uniform', activation = 'relu'))

        # Adding the output layer
        classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

        # Compiling the ANN
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Fitting the ANN to the Training set
        classifier.fit(X_resampled, y_resampled, batch_size=25, nb_epoch=200)

    elif clf_type == 2:
        # Grid search for finding the best hyperparameter
        Cs = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1]
        gammas = [0.001, 0.01, 0.1, 1]
        kernel = ['linear', 'rbf']

        from sklearn.svm import SVC

        param_grid = {'C': Cs, 'kernel': kernel, 'gamma': gammas}
        gs = GridSearchCV(SVC(), param_grid, cv=5)
        gs.fit(X_resampled, y_resampled)
        dic = gs.best_params_
        best_C = dic['C']
        best_kernel = dic['kernel']
        best_gamma = dic['gamma']
        print("Chosen hyperparameters are:", gs.best_params_)

        from sklearn.svm import SVC
        classifier = SVC(C=best_C, kernel=best_kernel, gamma=best_gamma)
        classifier.fit(X_resampled, y_resampled)

    print("Exiting Classifier.......")
    return classifier


def predictor(X_test, classifier, y_test='', testing=False):
    if testing is False:
        y_pred = classifier.predict(X_test[:, 0:8])
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        from sklearn.metrics import accuracy_score
        ac = accuracy_score(y_test, y_pred)
        print(ac)

        messagebox.showinfo("Success", "Accuracy: {} \n Confusion matrix:\n {}".format(ac, cm))
    else:

        # Loading saved model 
        # loaded_model = pickle.load(open(filename,'rb'))
        sc = StandardScaler()
        y_pred = classifier.predict(sc.fit_transform(X_test[:, 0:8]))
        # y_pred = loaded_model.predict(X_test[:, 0:8])

    row_dict = {x: 0 for x in range(1, 7)}
    col_dict = {x: 0 for x in range(7, 13)}

    for i in range(len(y_pred)):
        # print(y_pred[i])

        if (y_pred[i] == 1):
            marker = int(X_test[i, 8])
            if (marker < 7):
                row_dict[marker] = row_dict.get(marker, 0) + 1
                # print(row_dict[marker])
            else:
                col_dict[marker] = col_dict.get(marker, 0) + 1
                # print(col_dict[marker])
    predicted_row = max(row_dict.items(), key=operator.itemgetter(1))[0]
    predicted_column = max(col_dict.items(), key=operator.itemgetter(1))[0]
    return predicted_row, predicted_column

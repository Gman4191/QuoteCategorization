from data_preprocessing import DataProcessor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

data_processor = DataProcessor()
df = data_processor.format_csv('data/data.csv', [])
features, encoded_labels, processed_df = data_processor.process_data(df)

features_train, features_test, labels_train, labels_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=69)

param_dist = {
    'alpha': [.1, .2, .3, .4, .5, .6, .7, .8, .9 , 1.0],
    'force_alpha' : [True, False],
    'norm' : [True, False]
    # Add more parameters and distributions
}

# svc_model = SVC(kernel='linear')
# svc_model.fit(features_train, labels_train)
# svc_labels_prediction = svc_model.predict(features_test)



# rfmodel = RandomForestClassifier()
# rfmodel.fit(features_train, labels_train)
# rf_labels_prediction = rfmodel.predict(features_test)

# svc_accuracy = accuracy_score(labels_test, svc_labels_prediction)

# rf_accuracy = accuracy_score(labels_test, rf_labels_prediction)
# print("Random Forest:" , rf_accuracy)
# print("Support Vector Machine:", svc_accuracy)

nb_model = ComplementNB(norm=True, force_alpha=True, alpha=1.0)
nb_model.fit(features_train.toarray(), labels_train)
nb_labels_prediction = nb_model.predict(features_test.toarray())
# nb_accuracy = accuracy_score(labels_test, nb_labels_prediction)
# print("Complement Naive Bayes:" , nb_accuracy)

random_search = RandomizedSearchCV(estimator=nb_model, param_distributions=param_dist, n_iter=50, scoring='accuracy', cv=5)
random_search.fit(features_train, labels_train)

best_params = random_search.best_params_
best_score = random_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)
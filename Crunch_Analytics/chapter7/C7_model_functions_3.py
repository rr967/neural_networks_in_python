import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def load_and_preprocess_data(csv_name):
    
    #import data 
    path = str('data/' + csv_name + '.csv')
    app_df = pd.read_csv(path, index_col = 0)
    
    #only keep new data from current month
    app_df = app_df.loc[app_df['month'] == csv_name.split('_')[1]]
    
    print ('training on month: ' + app_df.month.unique())
    print ('number of rows: ' + str(len(app_df)))
    #print 'value counts ' + str(app_df.churn.value_counts())
    
    app_df = app_df.drop(['month'], axis = 1) 
    
    #drop userid cause it's not important
    userId = app_df['userId']
    app_df = app_df.drop(['userId'], axis = 1)
    
    #put sub_amount on same scale
    def sub_amount_transformer(x):
        if x['payment_freq'] == 'Quarterly':
            return x['sub_amount'] * 4
        elif x['payment_freq'] == 'Monthly':
            return x['sub_amount'] * 12
        else:
            return x['sub_amount']

    app_df['sub_amount'] = app_df.apply(lambda x: sub_amount_transformer(x), axis = 1)
    
    #add dummy variables for categorical variables
    app_df = pd.get_dummies(data = app_df, columns = ['payment_method', 'payment_freq', 'platform'])

    #dummies for location
    app_df['location_be'] = [1 if x == 0 else 0 for x in app_df['location']]
    app_df.rename(columns={'location': 'location_nl'}, inplace=True)
    
    #make y, X variables
    y = app_df['churn']
    X = app_df.drop(["churn"], axis=1)
    
    return X, y

def make_churn_model(csv_name):
    
    X, y = load_and_preprocess_data(csv_name)
    
    # Split the data into test and training (30% for test)
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.25,
                                                        stratify = y,
                                                        random_state = 123)
    
    rf = RandomForestClassifier(n_estimators=100)

    # Train the classifier using the train data
    rf = rf.fit(X_train, Y_train)

    # Validate the classifier
    accuracy = rf.score(X_test, Y_test)

    # Make a confusion matrix
    prediction = rf.predict(X_test)

    conf_matrix = pd.DataFrame(
        confusion_matrix(Y_test, prediction), 
        columns=["Predicted False", "Predicted True"], 
        index=["Actual False", "Actual True"]
    )

    return rf, accuracy, conf_matrix

def test_model(csv_name, rf):
    
    X, y = load_and_preprocess_data(csv_name)
    
    # Validate the classifier
    accuracy = rf.score(X, y)

    # Make a confusion matrix
    prediction = rf.predict(X)

    conf_matrix = pd.DataFrame(
        confusion_matrix(y, prediction), 
        columns=["Predicted False", "Predicted True"], 
        index=["Actual False", "Actual True"]
    )
    
    return accuracy, conf_matrix

def import_data_and_split_datasets(csv_name):
    
    #import data
    path = str('data/' + csv_name + '.csv')
    df = pd.read_csv(path, index_col = 0)
    current_month = csv_name.split('_')[1]
    
    #subset 2 dataframes
    past_df = df.loc[df['month'] != current_month]
    new_df = df.loc[df['month'] == current_month]
    
    return past_df, new_df

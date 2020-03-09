from django.shortcuts import render
from requests import request
from django.http import HttpResponse, JsonResponse
from csv import DictWriter, DictReader
from datetime import datetime
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, roc_auc_score
import pickle


def get_response(url):
    if not url:
        return None
    response = request(method='GET', url=url).json()
    return response


def parse_date(typed_at):
    dt = datetime.strptime(typed_at, '%Y-%m-%dT%H:%M:%S.%f')
    return dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.microsecond


def save_data_in_csv_file(output_filename, data, headers, user, mode='a'):
    valid_string = 'Be Authentic. Be Yourself. Be Typing.'
    user = user[5:-5]
    with open(output_filename, mode=mode) as in_csv:
        writer = DictWriter(in_csv, fieldnames=headers)
        if mode == 'w':
            writer.writeheader()
        for rows in data['user_data']:
            if not validate_input(rows):
                continue
            l = 0
            for row in rows:
                if row['character'] == valid_string[l]:
                    row['user'] = user
                    row['year'], row['month'], row['day'], row['hour'], row['minute'], \
                    row['second'], row['microsecond'] = parse_date(row['typed_at'])
                    writer.writerow(row)
                    l += 1


def download_all_users_data_into_csv(base_url, output_filename, headers):
    if not base_url:
        base_url = ''
    first_user = base_url.split('/')[-1]
    users = {first_user}
    base_url = '/'.join(base_url.split('/')[:-1])
    data = get_response(os.path.join(base_url, first_user))
    if data:
        save_data_in_csv_file(output_filename, data, headers, first_user, mode='w')
        next_user = data.get('next')
        users.add(next_user)
        while next_user:
            next_url = os.path.join(base_url, next_user)
            data = get_response(next_url)
            if not data:
                next_url = False
                continue
            save_data_in_csv_file(output_filename, data, headers, next_user)
            users.add(next_user)
            next_user = data.get('next')
        return users


def validate_input(record):
    valid_string = 'Be Authentic. Be Yourself. Be Typing.'
    valid_string_set = set(valid_string)
    user_string = ''
    l = 0
    for r in record:
        if r['character'] not in valid_string_set:
            continue
        if r['character'] != valid_string[l]:
            continue
        user_string += r['character']
        l += 1
    return valid_string == user_string


def data_pipeline(datafile, test_set_size):
    """
    Feature engineering
    """
    if not test_set_size or test_set_size <= 0:
        test_set_size = 0.05
    features = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'character']
    labels = ['user']
    df = pd.read_csv(datafile)
    categories = {c: idx for idx, c in enumerate('Be Authentic. Be Yourself. Be Typing.')}
    df.replace({"character": categories}, inplace=True)
    X, y = df[features], df[labels]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_size, random_state=101)
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    # roc_auc = auc(y_true, y_pred)
    # roc_auc = roc_auc_score(y_true, y_prob[:, -1])
    return accuracy, precision, recall, f1, 0


def save_model(clf, pkl_filename):
    with open(pkl_filename, 'wb') as fp:
        pickle.dump(clf, fp)


def adaboost_clf(X_train, X_test, y_train, y_test, pkl_filename):
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    save_model(clf, pkl_filename)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)
    return evaluate(y_test, y_pred, y_prob)


def train_model(request):
    training_data_url = request.GET.get('training_data_url')
    test_set_size = float(request.GET.get('test_set_size', 0))
    pkl_filename = request.GET.get('model_filename')
    if not training_data_url:
        return render(request, 'Train/ModelTraining.html', dict())
    first_user = training_data_url.split('/')[-1][5:-5]
    output_filename = 'training_data_{}.csv'.format(first_user)
    headers = ['typed_at', 'character', 'user', 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
    if not os.path.exists(output_filename):
        download_all_users_data_into_csv(training_data_url, output_filename, headers)
    X_train, X_test, y_train, y_test = data_pipeline(output_filename, test_set_size)
    if not os.path.exists(pkl_filename):
        accuracy, precision, recall, f1, roc_auc = adaboost_clf(X_train, X_test, y_train, y_test, pkl_filename)

    pkl_filenames = [m for m in os.listdir(os.getcwd()) if '.pkl' in m]
    evaluation = list()
    for pkl_filename in pkl_filenames:
        with open(pkl_filename, 'rb') as file:
            clf = pickle.load(file)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
        accuracy, precision, recall, f1, roc_auc = evaluate(y_test, y_pred, y_prob)
        model = pkl_filename.split('.pkl')[0]
        evaluation.append({'model': model.title(),  'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc})
    context = {'listings': evaluation}
    if training_data_url and test_set_size:
        return JsonResponse(context)
    return render(request, 'Train/ModelTraining.html', context)


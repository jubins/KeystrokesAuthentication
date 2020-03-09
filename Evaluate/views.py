from django.shortcuts import render
from requests import request
from django.http import HttpResponse, JsonResponse
from csv import DictWriter, DictReader
from datetime import datetime
import os
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


def save_data_in_csv_file(output_filename, data, headers, user='', mode='a'):
    valid_string = 'Be Authentic. Be Yourself. Be Typing.'
    with open(output_filename, mode=mode) as in_csv:
        writer = DictWriter(in_csv, fieldnames=headers)
        if mode == 'w':
            writer.writeheader()
        for rows in data['attempts']:
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
    data = get_response(base_url)
    if data:
        save_data_in_csv_file(output_filename, data, headers, mode='w')
        next_user = data.get('next')
        while next_user:
            next_url = os.path.join(base_url, next_user)
            data = get_response(next_url)
            if not data:
                next_url = False
                continue
            save_data_in_csv_file(output_filename, data, headers)
            next_user = data.get('next')


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
    features = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond', 'character']
    labels = ['user']
    df = pd.read_csv(datafile)
    categories = {c: idx for idx, c in enumerate('Be Authentic. Be Yourself. Be Typing.')}
    categories_rev = {idx: c for c, idx in categories.items()}
    df.replace({"character": categories}, inplace=True)
    X, y = df[features], df[labels]
    if not test_set_size:
        return X, y, categories_rev, df['typed_at']
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


def load_model(pkl_filename):
    if not pkl_filename:
        return None
    with open(pkl_filename, 'rb') as fp:
        clf = pickle.load(fp)
    return clf


def read_csv(output_filename, headers, first_n):
    context = list()
    with open(output_filename, mode='r') as in_csv:
        reader = DictReader(in_csv)
        for idx, row in enumerate(reader):
            d = {k: row.get(k) for k in headers}
            d['id'] = idx+1
            context.append(d)
            if first_n == 1:
                break
            first_n -= 1
    return context


def correct_csv(filename, df_typed_at, categories_rev, headers):
    output_filename = filename[:-5]+'.csv'
    with open(output_filename, mode='w', newline='') as out_csv:
        writer = DictWriter(out_csv, fieldnames=headers)
        writer.writeheader()
        with open(filename, mode='r') as in_csv:
            reader = DictReader(in_csv)
            for i, row in enumerate(reader):
                del row['']
                row['typed_at'] = df_typed_at[i]
                row['character'] = categories_rev[int(row['character'])]
                writer.writerow(row)
    os.remove(filename)


def evaluate_model(request):
    test_data_url = request.GET.get('test_data_url')
    test_model_id = request.GET.get('test_data_id')
    export_csv = request.GET.get('export')
    if export_csv:
        return export_as_csv(request)
    else:
        print(test_model_id)
        first_n = int(request.GET.get('first_n', 0))
        pkl_filenames = [m for m in os.listdir(os.getcwd()) if '.pkl' in m]
        if not test_data_url:
            return render(request, 'Evaluate/ModelEvaluation.html', {'models': pkl_filenames})
        test_user = test_data_url.split('/')[-1][:-5]
        output_filename = 'test_data_{}.csv'.format(test_user)
        headers = ['typed_at', 'character', 'user', 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
        if not os.path.exists(output_filename):
            download_all_users_data_into_csv(test_data_url, output_filename, headers)
        X_test, y_test, categories_rev, df_typed_at = data_pipeline(output_filename, test_set_size=None)
        df_typed_at = list(df_typed_at.values)
        test_model_id = pkl_filenames[0]
        clf = load_model(test_model_id)
        if clf:
            y_pred = clf.predict(X_test)
            X_test['user'] = y_pred
            temp_output = output_filename[:-4]+'_Output1.csv'
            X_test.to_csv(temp_output)
            correct_csv(temp_output, df_typed_at, categories_rev, headers)
            temp_output = output_filename[:-4]+'_Output.csv'
            evaluation = read_csv(temp_output, headers, first_n=first_n)
            context = {'listings': evaluation}
            if test_data_url:
                return JsonResponse(context)
        return render(request, 'Evaluate/ModelEvaluation.html', {'models': pkl_filenames})


def export_as_csv(request):
    test_data_url = request.GET.get('test_data_url')
    test_user = test_data_url.split('/')[-1][:-5]
    output_filename = 'test_data_{}.csv'.format(test_user)
    temp_output = output_filename[:-4]+'_Output.csv'
    data = open(temp_output, 'r').read()
    response = HttpResponse(data, content_type='application/csv')
    response['Content-Disposition'] = 'attachment; filename="{}"'.format(temp_output)
    return JsonResponse({'listings': data})

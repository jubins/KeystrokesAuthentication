from django.shortcuts import render
from requests import request
from django.http import HttpResponse, JsonResponse
from csv import DictWriter, DictReader
from datetime import datetime
from Train.models import TraininData
import os


def get_response(url):
    if not url:
        return None
    print(url)
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


def analyze_data(request):
    training_data_url = request.GET.get('training_data_url')
    first_n = int(request.GET.get('first_n', 50))
    if not training_data_url:
        return render(request, 'Analyze/TrainingDataAnalysis.html', dict())
    first_user = training_data_url.split('/')[-1][5:-5]
    output_filename = 'training_data_{}.csv'.format(first_user)
    headers = ['typed_at', 'character', 'user', 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
    if not os.path.exists(output_filename):
        download_all_users_data_into_csv(training_data_url, output_filename, headers)
    data = read_csv(output_filename, headers, first_n=first_n)
    context = {'listings': data}
    if training_data_url and first_n:
        return JsonResponse(context)
    return render(request, 'Analyze/TrainingDataAnalysis.html', context)


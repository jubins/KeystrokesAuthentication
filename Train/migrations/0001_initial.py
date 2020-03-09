# Generated by Django 3.0.4 on 2020-03-08 04:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TraininData',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user', models.CharField(max_length=120)),
                ('typed_at', models.CharField(max_length=120)),
                ('character', models.CharField(max_length=20)),
                ('year', models.CharField(max_length=20)),
                ('month', models.CharField(max_length=20)),
                ('day', models.CharField(max_length=20)),
                ('hour', models.CharField(max_length=20)),
                ('minute', models.CharField(max_length=20)),
                ('second', models.CharField(max_length=20)),
                ('microsecond', models.CharField(max_length=20)),
            ],
        ),
    ]
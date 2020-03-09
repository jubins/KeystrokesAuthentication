from django.db import models


class TraininData(models.Model):
    user = models.CharField(max_length=120)
    typed_at = models.CharField(max_length=120)
    character = models.CharField(max_length=20)
    year = models.CharField(max_length=20)
    month = models.CharField(max_length=20)
    day = models.CharField(max_length=20)
    hour = models.CharField(max_length=20)
    minute = models.CharField(max_length=20)
    second = models.CharField(max_length=20)
    microsecond = models.CharField(max_length=20)

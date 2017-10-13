# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0012_task4'),
    ]

    operations = [
        migrations.AddField(
            model_name='task4',
            name='movie_count',
            field=models.IntegerField(default=0),
        ),
    ]

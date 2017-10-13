# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='mlratings',
            name='norm_weight',
            field=models.CharField(db_index=True, max_length=256, blank=True),
        ),
        migrations.AddField(
            model_name='mltags',
            name='norm_weight',
            field=models.CharField(db_index=True, max_length=256, blank=True),
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0004_auto_20170911_0555'),
    ]

    operations = [
        migrations.AddField(
            model_name='mlratings',
            name='norm_weight',
            field=models.FloatField(default=0.0, max_length=256, db_index=True, blank=True),
        ),
        migrations.AddField(
            model_name='mltags',
            name='norm_weight',
            field=models.FloatField(default=0.0, max_length=256, db_index=True, blank=True),
        ),
    ]

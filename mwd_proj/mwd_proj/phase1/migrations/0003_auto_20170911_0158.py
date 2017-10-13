# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0002_auto_20170911_0155'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mlratings',
            name='norm_weight',
            field=models.CharField(default='', max_length=256, db_index=True, blank=True),
        ),
        migrations.AlterField(
            model_name='mltags',
            name='norm_weight',
            field=models.CharField(default='', max_length=256, db_index=True, blank=True),
        ),
    ]

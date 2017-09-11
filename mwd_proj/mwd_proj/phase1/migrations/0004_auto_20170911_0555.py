# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0003_auto_20170911_0158'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='mlratings',
            name='norm_weight',
        ),
        migrations.RemoveField(
            model_name='mltags',
            name='norm_weight',
        ),
        migrations.AlterField(
            model_name='mlratings',
            name='timestamp',
            field=models.BigIntegerField(blank=True),
        ),
        migrations.AlterField(
            model_name='mltags',
            name='timestamp',
            field=models.BigIntegerField(blank=True),
        ),
    ]

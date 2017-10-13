# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase2', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mltags',
            name='userid',
            field=models.IntegerField(null=True),
        ),
    ]

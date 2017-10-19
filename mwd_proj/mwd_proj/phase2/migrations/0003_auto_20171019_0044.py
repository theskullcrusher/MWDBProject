# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase2', '0002_auto_20171013_0628'),
    ]

    operations = [
        migrations.AlterField(
            model_name='mltags',
            name='userid',
            field=models.ForeignKey(to='phase2.MlUsers', null=True),
        ),
    ]

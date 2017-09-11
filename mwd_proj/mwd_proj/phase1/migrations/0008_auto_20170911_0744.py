# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0007_task1'),
    ]

    operations = [
        migrations.AlterField(
            model_name='task1',
            name='tagid',
            field=models.TextField(default='', null=True),
        ),
    ]

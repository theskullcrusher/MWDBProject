# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0008_auto_20170911_0744'),
    ]

    operations = [
        migrations.RenameField(
            model_name='task1',
            old_name='tagid',
            new_name='tag',
        ),
    ]

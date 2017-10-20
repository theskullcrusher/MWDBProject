# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase2', '0004_task5'),
    ]

    operations = [
        migrations.CreateModel(
            name='Task6',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('movieid', models.IntegerField(blank=True)),
                ('tagid', models.IntegerField(blank=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
    ]

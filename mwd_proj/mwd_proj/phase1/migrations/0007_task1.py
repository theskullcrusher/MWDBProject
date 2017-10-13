# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase1', '0006_auto_20170911_0605'),
    ]

    operations = [
        migrations.CreateModel(
            name='Task1',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('actorid', models.IntegerField(null=True)),
                ('tagid', models.IntegerField(null=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
    ]

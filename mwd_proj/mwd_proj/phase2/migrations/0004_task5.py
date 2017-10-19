# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('phase2', '0003_auto_20171019_0044'),
    ]

    operations = [
        migrations.CreateModel(
            name='Task5',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('genre', models.TextField(default='', null=True)),
                ('actorid', models.IntegerField(blank=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
    ]

# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='GenomeTags',
            fields=[
                ('tagid', models.IntegerField(serialize=False, primary_key=True)),
                ('tag', models.CharField(max_length=256, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='ImdbActorInfo',
            fields=[
                ('actorid', models.IntegerField(serialize=False, primary_key=True)),
                ('name', models.CharField(max_length=512, blank=True)),
                ('gender', models.CharField(max_length=56, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='MlMovies',
            fields=[
                ('movieid', models.IntegerField(serialize=False, primary_key=True)),
                ('moviename', models.TextField(blank=True)),
                ('genres', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='MlRatings',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('movieid', models.IntegerField()),
                ('userid', models.IntegerField()),
                ('imdbid', models.IntegerField(blank=True)),
                ('rating', models.IntegerField(blank=True)),
                ('timestamp', models.CharField(max_length=256, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='MlTags',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('userid', models.IntegerField()),
                ('movieid', models.IntegerField()),
                ('tagid', models.IntegerField()),
                ('timestamp', models.CharField(max_length=256, blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='MlUsers',
            fields=[
                ('userid', models.IntegerField(serialize=False, primary_key=True)),
            ],
        ),
        migrations.CreateModel(
            name='MovieActor',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('movieid', models.IntegerField()),
                ('actorid', models.IntegerField()),
                ('actor_movie_rank', models.IntegerField()),
            ],
        ),
    ]

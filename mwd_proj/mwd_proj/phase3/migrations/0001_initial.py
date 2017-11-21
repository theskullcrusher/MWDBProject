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
                ('year', models.IntegerField(blank=True)),
                ('genres', models.TextField(blank=True)),
            ],
        ),
        migrations.CreateModel(
            name='MlRatings',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('imdbid', models.IntegerField(blank=True)),
                ('rating', models.IntegerField(blank=True)),
                ('timestamp', models.BigIntegerField(blank=True)),
                ('norm_weight', models.FloatField(default=0.0, db_index=True, blank=True)),
                ('movieid', models.ForeignKey(to='phase3.MlMovies')),
            ],
        ),
        migrations.CreateModel(
            name='MlTags',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('timestamp', models.BigIntegerField(blank=True)),
                ('norm_weight', models.FloatField(default=0.0, db_index=True, blank=True)),
                ('movieid', models.ForeignKey(to='phase3.MlMovies')),
                ('tagid', models.ForeignKey(to='phase3.GenomeTags')),
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
                ('actor_movie_rank', models.IntegerField()),
                ('norm_rank', models.FloatField(default=0.0, db_index=True, blank=True)),
                ('actorid', models.ForeignKey(to='phase3.ImdbActorInfo')),
                ('movieid', models.ForeignKey(to='phase3.MlMovies')),
            ],
        ),
        migrations.CreateModel(
            name='Task1',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('actorid', models.IntegerField(null=True)),
                ('tag', models.TextField(default='', null=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task2',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('genre', models.TextField(default='', null=True)),
                ('tag', models.TextField(default='', null=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task3',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('userid', models.IntegerField(null=True)),
                ('tag', models.TextField(default='', null=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task4',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('genre', models.TextField(default='', null=True)),
                ('tag', models.TextField(default='', null=True)),
                ('score', models.FloatField(null=True)),
                ('movie_count', models.IntegerField(default=0)),
            ],
        ),
        migrations.CreateModel(
            name='Task5',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('genre', models.TextField(default='', null=True)),
                ('actorid', models.IntegerField(blank=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task6',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('movieid', models.IntegerField(blank=True)),
                ('tagid', models.IntegerField(blank=True)),
                ('score', models.FloatField(null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Task7',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('movieid', models.IntegerField(blank=True)),
                ('tagid', models.IntegerField(blank=True)),
                ('rating', models.IntegerField(null=True)),
            ],
        ),
        migrations.AddField(
            model_name='mltags',
            name='userid',
            field=models.ForeignKey(to='phase3.MlUsers', null=True),
        ),
        migrations.AddField(
            model_name='mlratings',
            name='userid',
            field=models.ForeignKey(to='phase3.MlUsers'),
        ),
    ]

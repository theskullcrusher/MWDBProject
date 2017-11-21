from __future__ import unicode_literals
from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
from django.contrib.postgres.fields import ArrayField

class GenomeTags(models.Model):
	tagid = models.IntegerField(primary_key=True)
	tag = models.CharField(max_length=256, blank=True)

class ImdbActorInfo(models.Model):
	actorid = models.IntegerField(primary_key=True)
	name = models.CharField(max_length=512, blank=True)
	gender = models.CharField(max_length=56, blank=True)

class MlMovies(models.Model):
	movieid = models.IntegerField(primary_key=True)
	moviename = models.TextField(blank=True)
	year = models.IntegerField(blank=True)
	genres = models.TextField(blank=True)

class MlUsers(models.Model):
	userid = models.IntegerField(primary_key=True)

class MovieActor(models.Model):
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	actorid = models.ForeignKey('ImdbActorInfo', on_delete=models.CASCADE)
	actor_movie_rank = models.IntegerField()
	norm_rank = models.FloatField(blank=True, db_index=True, default=0.0)

class MlTags(models.Model):
	userid = models.ForeignKey('MlUsers', on_delete=models.CASCADE, null=True)
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	tagid = models.ForeignKey('GenomeTags', on_delete=models.CASCADE)
	timestamp = models.BigIntegerField(blank=True)
	norm_weight = models.FloatField( blank=True, db_index=True, default=0.0)

class MlRatings(models.Model):
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	userid = models.ForeignKey('MlUsers', on_delete=models.CASCADE)
	imdbid = models.IntegerField(blank=True)
	rating = models.IntegerField(blank=True)
	timestamp = models.BigIntegerField( blank=True)
	norm_weight = models.FloatField(blank=True, db_index=True, default=0.0)

class Task1(models.Model):
	actorid = models.IntegerField(null=True)
	tag = models.TextField(null=True, default='')
	score = models.FloatField(null=True)

class Task2(models.Model):
	genre = models.TextField(null=True, default='')
	tag = models.TextField(null=True, default='')
	score = models.FloatField(null=True)

class Task3(models.Model):
	userid = models.IntegerField(null=True)
	tag = models.TextField(null=True, default='')
	score = models.FloatField(null=True)

class Task4(models.Model):
	genre = models.TextField(null=True, default='')
	tag = models.TextField(null=True, default='')
	score = models.FloatField(null=True)
	movie_count = models.IntegerField(default=0)

class Task5(models.Model):
	genre = models.TextField(null=True, default='')
	actorid = models.IntegerField(blank=True)
	score = models.FloatField(null=True)

class Task6(models.Model):
	movieid = models.IntegerField(blank=True)
	tagid = models.IntegerField(blank=True)
	score = models.FloatField(null=True)

class Task7(models.Model):
	movieid = models.IntegerField(blank=True)
	tagid = models.IntegerField(blank=True)
	rating = models.IntegerField(null=True)
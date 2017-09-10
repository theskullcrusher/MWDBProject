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
	genres = ArrayField(models.CharField(max_length=512,blank=True))

class MlUsers(models.Model):
	userid = models.IntegerField(primary_key=True)

class MovieActor(models.Model):
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	actorid = models.ForeignKey('ImdbActorInfo', on_delete=models.CASCADE)
	actor_movie_rank = models.IntegerField()

class MlTags(models.Model):
	userid = models.ForeignKey('MlUsers', on_delete=models.CASCADE)
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	tagid = models.ForeignKey('GenomeTags', on_delete=models.CASCADE)
	timestamp = models.CharField(max_length=256, blank=True)

class MlRatings(models.Model):
	movieid = models.ForeignKey('MlMovies', on_delete=models.CASCADE)
	userid = models.ForeignKey('MlUsers', on_delete=models.CASCADE)
	imdbid = models.IntegerField(blank=True)
	rating = models.IntegerField(blank=True)
	timestamp = models.CharField(max_length=256, blank=True)

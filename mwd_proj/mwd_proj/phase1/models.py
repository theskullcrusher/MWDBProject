import unicode_literals
from django.db import models
from django.contrib.auth.models import User
from datetime import datetime
from django.contrib.auth.signals import user_logged_in, user_logged_out, user_login_failed
from django.dispatch import receiver


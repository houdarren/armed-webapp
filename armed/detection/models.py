from django.core.files.storage import FileSystemStorage
from django.db import models

input_fs = FileSystemStorage(location='/media/input')
output_fs = FileSystemStorage(location='/media/output')

class InputImage(models.Model):
	name = models.CharField(max_length=200)
	image = models.ImageField(storage=input_fs)

class OutputImage(models.Model):
	name = models.CharField(max_length=200)
	image = models.ImageField(storage=output_fs)
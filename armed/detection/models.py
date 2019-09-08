from django.core.files.storage import FileSystemStorage
from django.db import models

input_fs = 'images/'
output_fs = 'images/output/'

class InputImage(models.Model):
	name = models.TextField(max_length=255)
	image = models.ImageField(upload_to=input_fs)

	def __str__(self):
		return self.name

class OutputImage(models.Model):
	name = models.TextField(max_length=255)
	image = models.ImageField(upload_to=input_fs)

	def __str__(self):
		return self.name
from django.contrib import admin

from .models import InputImage
from .models import OutputImage


admin.site.register(InputImage)

admin.site.register(OutputImage)
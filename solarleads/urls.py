from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.views.static import serve as media_serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('homes.urls')),
    # Serve generated satellite/annotation images in any environment.
    # Fine for a low-traffic demo; use object storage / a CDN at scale.
    re_path(r'^media/(?P<path>.*)$', media_serve, {'document_root': settings.MEDIA_ROOT}),
]

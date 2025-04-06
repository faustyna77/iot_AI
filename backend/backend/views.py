from django.contrib.auth.decorators import login_required
from rest_framework_simplejwt.tokens import RefreshToken
from django.shortcuts import redirect
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required



@login_required
def google_redirect(request):
    user_email = request.user.email
    streamlit_url = f"http://localhost:8501/?token={user_email}"
    return redirect(streamlit_url)

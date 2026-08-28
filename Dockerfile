FROM python:3.10-slim

# Hugging Face Spaces requires running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .

# Install dependencies (CPU ONLY for PyTorch to save 800MB of space!)
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user . .

# Collect static files for production
RUN python manage.py collectstatic --noinput

# Hugging Face exposes port 7860 by default
EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "dlserver.wsgi:application"]

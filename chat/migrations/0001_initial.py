# Generated by Django 4.1.3 on 2023-05-04 06:56

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Chat",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("user_input", models.CharField(max_length=200)),
                ("bot_response", models.CharField(max_length=200)),
            ],
        ),
    ]

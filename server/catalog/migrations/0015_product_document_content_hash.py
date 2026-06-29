# Generated for YAZ-10: add a canonical content_hash to Product and Document so the
# sync layer can detect no-op re-syncs and skip the expensive re-index.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("catalog", "0014_pgvector_ann_indexes"),
    ]

    operations = [
        migrations.AddField(
            model_name="product",
            name="content_hash",
            field=models.CharField(blank=True, max_length=64, null=True),
        ),
        migrations.AddField(
            model_name="document",
            name="content_hash",
            field=models.CharField(blank=True, max_length=64, null=True),
        ),
    ]

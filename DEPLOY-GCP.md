# Deploying Solar Lead Finder to Google Cloud Run

This app ships as a single container ([`Dockerfile`](Dockerfile)) and runs great on
**Cloud Run**: it scales to zero (≈$0 when idle), needs no GPU (ONNX Runtime on CPU),
and keeps billing next to the Google Maps API it already uses.

Two things the container filesystem **cannot** keep (it is wiped on every redeploy),
so both move to managed services — the app switches to them automatically when their
env vars are set, and stays on SQLite + local files when they are not:

| Concern | Local / demo default | Production |
| --- | --- | --- |
| Database (leads, confirmations) | SQLite | **managed Postgres** via `DATABASE_URL` |
| Media (annotated rooftop images) | local `media/` | **Google Cloud Storage** via `GS_BUCKET_NAME` |

Static files are served by WhiteNoise from inside the image — nothing to configure.

---

## 0. Prerequisites

- `gcloud` CLI installed and logged in: `gcloud auth login`
- A Google Cloud **project with billing enabled** (create it in the console).
- You are `Owner` on the project (avoids IAM friction on the first deploy).

You do **not** need Docker locally — Cloud Build builds the image from the `Dockerfile`.

---

## 1. Set variables

```bash
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export SERVICE="solar-lead-finder"
export BUCKET="${PROJECT_ID}-solar-media"      # must be globally unique

gcloud config set project "$PROJECT_ID"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"   # Cloud Run's default service account
```

## 2. Enable APIs

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  storage.googleapis.com \
  secretmanager.googleapis.com
```

## 3. Create the media bucket (GCS)

```bash
gcloud storage buckets create "gs://${BUCKET}" \
  --location="$REGION" \
  --uniform-bucket-level-access

# Public read so <img> tags resolve (the app serves media with querystring_auth off).
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="allUsers" --role="roles/storage.objectViewer"

# Let the Cloud Run service read/write rooftop images.
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/storage.objectAdmin"
```

> Prefer not to make the bucket public? Remove the `allUsers` binding and set
> `querystring_auth` back on in `settings.py` so the app serves signed URLs instead.

## 4. Provision Postgres — pick ONE

### Option A — External Postgres (simplest, free tier available)

Create a database on **Neon** or **Supabase**, copy its connection string, and:

```bash
export DATABASE_URL="postgresql://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require"
```

### Option B — Cloud SQL for PostgreSQL (GCP-native; ~$9+/mo, no scale-to-zero)

```bash
gcloud services enable sqladmin.googleapis.com
gcloud sql instances create solar-pg --database-version=POSTGRES_16 \
  --tier=db-f1-micro --region="$REGION"
gcloud sql databases create solarleads --instance=solar-pg
gcloud sql users create solaruser --instance=solar-pg --password="CHOOSE_A_STRONG_PASSWORD"

INSTANCE_CONN="$(gcloud sql instances describe solar-pg --format='value(connectionName)')"
# Unix-socket URL (the socket is mounted by --add-cloudsql-instances at deploy time):
export DATABASE_URL="postgresql://solaruser:CHOOSE_A_STRONG_PASSWORD@/solarleads?host=/cloudsql/${INSTANCE_CONN}"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${RUNTIME_SA}" --role="roles/cloudsql.client"
```

## 5. Store secrets in Secret Manager

```bash
# A fresh Django secret key:
export SECRET_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(50))')"
export GMAPS_KEY="your-google-maps-static-api-key"   # omit to run the demo on mock data

printf '%s' "$SECRET_KEY"    | gcloud secrets create solar-secret-key   --data-file=-
printf '%s' "$DATABASE_URL"  | gcloud secrets create solar-database-url --data-file=-
printf '%s' "$GMAPS_KEY"     | gcloud secrets create solar-gmaps-key    --data-file=-

# Let the Cloud Run service read them.
for S in solar-secret-key solar-database-url solar-gmaps-key; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${RUNTIME_SA}" --role="roles/secretmanager.secretAccessor"
done
```

> Rotating a value later: `printf '%s' "$NEW" | gcloud secrets versions add <name> --data-file=-`

## 6. Deploy

```bash
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --memory 1Gi --cpu 1 \
  --concurrency 4 --timeout 300 \
  --min-instances 0 --max-instances 2 \
  --set-env-vars "DJANGO_DEBUG=False,GS_BUCKET_NAME=${BUCKET},GS_PROJECT_ID=${PROJECT_ID}" \
  --set-secrets "SECRET_KEY=solar-secret-key:latest,DATABASE_URL=solar-database-url:latest,GOOGLE_STREET_VIEW_API_KEY=solar-gmaps-key:latest"
```

For **Cloud SQL (Option B)** add the socket mount:

```bash
  --add-cloudsql-instances "$INSTANCE_CONN"
```

Cloud Run sets `K_SERVICE` automatically, so `settings.py` already trusts the
`*.run.app` host and CSRF origin — no `ALLOWED_HOSTS` wiring needed. The container
entrypoint runs `migrate` and seeds/scans the demo homes on first boot.

## 7. Verify

```bash
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
```

Open that URL, search a city (e.g. `San Francisco, CA`), open a home, and confirm the
annotated rooftop image loads from `storage.googleapis.com/${BUCKET}/...`. Tail logs with:

```bash
gcloud run services logs read "$SERVICE" --region "$REGION" --limit 100
```

---

## Notes & caveats

- **Custom domain.** Map one with `gcloud run domain-mappings create`, then add it to
  the service's env: `ALLOWED_HOSTS=yourdomain.com` and
  `CSRF_TRUSTED_ORIGINS=https://yourdomain.com`.
- **Migrations run on every cold start** (in the entrypoint). Fine for this app; for
  heavy concurrent scaling, move `migrate` to a Cloud Run **Job** or a release step and
  drop it from the entrypoint.
- **Stage-5 reference fingerprints** (`media/solar_references/`, `media/nosolar_references/`)
  are gitignored and live only on your local disk. `gcloud run deploy --source .` uploads
  them from your working copy, so deploying from **this** checkout includes them; a deploy
  from a fresh `git clone` would not, and the detector would fall back to the ONNX
  classifier alone (still the primary model — it is force-tracked in git).
- **Cost.** Scale-to-zero means ~$0 when idle; you pay per request-second plus a few cents
  for GCS. Cloud SQL (Option B) has a monthly floor even when idle — Option A (Neon/Supabase
  free tier) avoids it.
- **Billing floor removed?** `--min-instances 0` is the default here; raise it to 1 only if
  cold-start latency bothers you (it brings back a small always-on cost).

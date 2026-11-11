# SeaweedFS

The SDK traces SeaweedFS through its S3 gateway (boto3, aiobotocore, aioboto3, s3fs) and
through the Filer HTTP API (`httpx`, `aiohttp`). SeaweedFS is recognised only on endpoints
you declare; nothing is guessed from a port number or a response header.

## Declare your endpoints

```bash
# Comma-separated host[:port] or URLs. A bare host matches any port; "*.suffix" hosts allowed.
export GENAI_OTEL_SEAWEEDFS_ENDPOINTS=seaweed-s3.example.internal:8333,seaweed-filer.example.internal:8888
```

When `GENAI_OTEL_MEDIA_STORE=seaweed`, the media-store endpoint is declared automatically.

## What you get

**S3 gateway spans** (`s3.<Operation>`) keep `db.system="s3"` and gain:

- `db.system.name="seaweedfs"` when the endpoint is declared;
- `server.port` (always, for any S3 endpoint);
- `aws.s3.bucket`, `aws.s3.key` (the legacy `s3.bucket` / `s3.object` are still emitted but
  deprecated) and `aws.request_id`.

**Filer spans** (`seaweedfs.filer.<operation>`, client kind) for requests to declared hosts:

| Attribute | Value |
|---|---|
| `db.system.name` | `seaweedfs` |
| `db.operation.name` | `upload`, `append`, `download`, `list`, `stat`, `delete`, `delete_recursive`, `move`, `copy`, `mkdir`, `tag`, `other` |
| `http.request.method`, `http.response.status_code` | as sent / received |
| `server.address`, `server.port` | declared host and port |
| `url.path.prefix` | first 2 path segments only (`GENAI_OTEL_SEAWEEDFS_PATH_PREFIX_SEGMENTS`) |
| `seaweedfs.request.size`, `seaweedfs.response.size` | from `Content-Length` when stated |

File contents, file names beyond the prefix, query values, headers and error text are never
recorded. A 404 on `stat` or `download` is not marked as an error.

`requests` is not patched (patching it can break OTLP HTTP exporters), so Filer calls made
with `requests` are not traced. Use `httpx` or `aiohttp`, or the S3 gateway.

## Async S3 clients

`aiobotocore`, `aioboto3` and `s3fs` calls are traced as `s3.<Operation>` spans exactly like
boto3 calls, when `aiobotocore` is installed.

## The SDK's own media uploads

Uploads made by the multimodal media store are not traced by default. Set
`GENAI_OTEL_MEDIA_STORE_TRACE_CALLS=true` to see them as nested `s3.*` spans.

## Filter examples

```
db.system.name = "seaweedfs"
name = "seaweedfs.filer.upload" and http.response.status_code >= 500
```
